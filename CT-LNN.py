import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary
from fvcore.nn import FlopCountAnalysis, flop_count_table
Tensor = torch.Tensor


# ============================================================
# ============================================================

@dataclass
class CTLNNConfig:
    # [Temperature_C, RH_percent, CO2_ppm]
    input_dim: int = 3


    hidden_dim: int = 64


    output_dim: int = 3

    mech_hidden_dim: int = 64

    readout_hidden_dim: int = 64


    ncp_interneurons: int = 32
    ncp_command_neurons: int = 16
    ncp_motor_neurons: int = 64
    ncp_density: float = 0.35
    ncp_seed: int = 2026


    min_tau: float = 0.05
    max_step_h: float = 1.0 / 6.0


    temp_ref_C: float = 2.0
    rh_ref_percent: float = 90.0
    co2_ref_ppm: float = 400.0

    dropout: float = 0.0


# ============================================================

# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SparseLinear(nn.Module):


    def __init__(
        self,
        in_features: int,
        out_features: int,
        density: float = 0.35,
        bias: bool = True,
        seed: int = 2026,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        generator = torch.Generator()
        generator.manual_seed(seed)

        mask = torch.rand(out_features, in_features, generator=generator) < density


        for i in range(out_features):
            if not mask[i].any():
                j = torch.randint(0, in_features, (1,), generator=generator).item()
                mask[i, j] = True

        self.register_buffer("mask", mask.float())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.mul_(self.mask)

        if self.bias is not None:
            fan_in = max(1, int(self.mask.sum(dim=1).float().mean().item()))
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)


# ============================================================

# ============================================================

class MechanismGuidedBranch(nn.Module):
    """

        f_mech(h, x) = base_direction(h)
                       * g_T(T)
                       * g_RH(RH)

        x[..., 0] = temperature_C
        x[..., 1] = RH_percent
        x[..., 2] = CO2_ppm
    """

    def __init__(self, cfg: CTLNNConfig):
        super().__init__()

        self.cfg = cfg
        H = cfg.hidden_dim


        self.base_direction = MLP(
            in_dim=H,
            hidden_dim=cfg.mech_hidden_dim,
            out_dim=H,
            dropout=cfg.dropout,
        )


        self.log_alpha_T = nn.Parameter(torch.zeros(H))


        self.log_k_RH = nn.Parameter(torch.zeros(H))
        self.theta_RH = nn.Parameter(torch.full((H,), cfg.rh_ref_percent))


        self.log_co2_scale = nn.Parameter(torch.full((H,), math.log(1000.0)))
        self.log_co2_power = nn.Parameter(torch.zeros(H))

        self.register_buffer("temp_ref_K", torch.tensor(cfg.temp_ref_C + 273.15))
        self.register_buffer("co2_ref", torch.tensor(cfg.co2_ref_ppm))

    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        T_C = x[..., 0:1]
        RH = x[..., 1:2]
        CO2 = x[..., 2:3]

        # --------------------------------------------------------
        # 1) Arrhenius-type temperature gate

        # --------------------------------------------------------
        T_K = torch.clamp(T_C + 273.15, min=250.0, max=330.0)

        alpha_T = F.softplus(self.log_alpha_T).view(1, -1)

        g_T = torch.exp(
            alpha_T * (1.0 / self.temp_ref_K - 1.0 / T_K)
        )

        g_T = torch.clamp(g_T, min=0.2, max=5.0)

        # --------------------------------------------------------
        # 2) RH suppressive gate

        # --------------------------------------------------------
        k_RH = F.softplus(self.log_k_RH).view(1, -1) / 20.0
        theta_RH = self.theta_RH.view(1, -1)

        g_RH = 0.2 + 0.8 * torch.sigmoid(
            k_RH * (theta_RH - RH)
        )

        # --------------------------------------------------------
        # 3) CO2 saturating inhibition gate

        # --------------------------------------------------------
        co2_excess = torch.relu(CO2 - self.co2_ref)

        co2_scale = F.softplus(self.log_co2_scale).view(1, -1) + 1e-6
        co2_power = F.softplus(self.log_co2_power).view(1, -1) + 1.0

        g_CO2 = 1.0 / (
            1.0 + torch.pow(co2_excess / co2_scale, co2_power)
        )

        g_CO2 = torch.clamp(g_CO2, min=0.05, max=1.0)

        # --------------------------------------------------------
        # --------------------------------------------------------
        env_gate = g_T * g_RH * g_CO2

        return self.base_direction(h) * env_gate


# ============================================================

# ============================================================

class NCPResidualBranch(nn.Module):
    """
    NCP-style residual compensation branch。

    """

    def __init__(self, cfg: CTLNNConfig):
        super().__init__()

        in_dim = cfg.hidden_dim + cfg.input_dim
        I = cfg.ncp_interneurons
        C = cfg.ncp_command_neurons
        M = cfg.ncp_motor_neurons
        H = cfg.hidden_dim
        d = cfg.ncp_density
        s = cfg.ncp_seed

        self.sensory_to_inter = SparseLinear(
            in_features=in_dim,
            out_features=I,
            density=d,
            seed=s + 1,
        )

        self.sensory_to_command = SparseLinear(
            in_features=in_dim,
            out_features=C,
            density=d,
            seed=s + 2,
        )

        self.inter_to_command = SparseLinear(
            in_features=I,
            out_features=C,
            density=d,
            seed=s + 3,
        )

        self.command_to_motor = SparseLinear(
            in_features=C,
            out_features=M,
            density=d,
            seed=s + 4,
        )

        self.motor_to_hidden = SparseLinear(
            in_features=M,
            out_features=H,
            density=d,
            seed=s + 5,
        )

        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        z = torch.cat([h, x], dim=-1)

        inter = torch.tanh(
            self.sensory_to_inter(z)
        )

        command = torch.tanh(
            self.sensory_to_command(z)
            + self.inter_to_command(inter)
        )

        motor = torch.tanh(
            self.command_to_motor(command)
        )

        residual = self.motor_to_hidden(motor)

        return self.out_scale * residual


# ============================================================

# ============================================================

class CTLNNDynamics(nn.Module):
    """

        dh/dt = -h / tau(h, x)
                + f_mech(h, x)
                + f_ncp(h, x)
                + f_input(h, x)

    """

    def __init__(self, cfg: CTLNNConfig):
        super().__init__()

        self.cfg = cfg
        H = cfg.hidden_dim
        D = cfg.input_dim

        self.tau_net = MLP(
            in_dim=H + D,
            hidden_dim=64,
            out_dim=H,
            dropout=cfg.dropout,
        )

        self.mechanism_branch = MechanismGuidedBranch(cfg)
        self.ncp_branch = NCPResidualBranch(cfg)

        self.input_drive = MLP(
            in_dim=H + D,
            hidden_dim=64,
            out_dim=H,
            dropout=cfg.dropout,
        )

        self.input_drive_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        hx = torch.cat([h, x], dim=-1)

        tau = F.softplus(self.tau_net(hx)) + self.cfg.min_tau

        liquid_decay = -h / tau
        mechanism = self.mechanism_branch(h, x)
        ncp_residual = self.ncp_branch(h, x)
        input_drive = self.input_drive_scale * self.input_drive(hx)

        dhdt = liquid_decay + mechanism + ncp_residual + input_drive

        return dhdt


# ============================================================

# ============================================================

class MultiTaskPredictionHead(nn.Module):
    """


        h(t) -> shared representation
             -> Firmness head
             -> SSC head
             -> TA head
    """

    def __init__(self, cfg: CTLNNConfig):
        super().__init__()

        H = cfg.hidden_dim
        R = cfg.readout_hidden_dim

        self.shared = nn.Sequential(
            nn.Linear(H, R),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
        )

        self.firmness_head = nn.Linear(R, 1)
        self.ssc_head = nn.Linear(R, 1)
        self.ta_head = nn.Linear(R, 1)

    def forward(self, h: Tensor) -> Tensor:
        original_shape = h.shape[:-1]

        h_flat = h.reshape(-1, h.shape[-1])

        z = self.shared(h_flat)

        y = torch.cat(
            [
                self.firmness_head(z),
                self.ssc_head(z),
                self.ta_head(z),
            ],
            dim=-1,
        )

        return y.reshape(*original_shape, 3)


# ============================================================

# ============================================================

class CTLNN(nn.Module):
    """
    CT-LNN for cold-chain fruit quality prediction.

    """

    def __init__(self, cfg: CTLNNConfig = CTLNNConfig()):
        super().__init__()

        self.cfg = cfg

        self.dynamics = CTLNNDynamics(cfg)
        self.readout = MultiTaskPredictionHead(cfg)

        self.env_h0_encoder = MLP(
            in_dim=cfg.input_dim,
            hidden_dim=64,
            out_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        )

        self.env_y_h0_encoder = MLP(
            in_dim=cfg.input_dim + cfg.output_dim,
            hidden_dim=64,
            out_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        )

        self.learned_h0 = nn.Parameter(torch.zeros(cfg.hidden_dim))

    # ------------------------------------------------------------

    # ------------------------------------------------------------

    def _prepare_time_inputs(
        self,
        env_times: Tensor,
        env_values: Tensor,
        query_times: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:

        if env_values.dim() == 2:
            env_values = env_values.unsqueeze(0)

        if env_values.dim() != 3:
            raise ValueError(
                "env_values must have shape [B, L, input_dim] or [L, input_dim]."
            )

        if env_values.size(-1) != self.cfg.input_dim:
            raise ValueError(
                f"Expected env_values last dimension = {self.cfg.input_dim}."
            )

        device = env_values.device
        dtype = env_values.dtype

        B, L, _ = env_values.shape

        env_times = env_times.to(device=device, dtype=dtype)
        query_times = query_times.to(device=device, dtype=dtype)

        if env_times.dim() == 2:
            if env_times.size(0) != B:
                raise ValueError(
                    "When env_times is [B, L], B must match env_values."
                )


            if not torch.allclose(
                env_times,
                env_times[0:1].expand_as(env_times),
                atol=1e-6,
            ):
                raise ValueError(
                    "env_times must be shared across batch. "
                    "Please resample or pad if each sample has a different time grid."
                )

            env_times = env_times[0]

        elif env_times.dim() != 1:
            raise ValueError("env_times must be [L] or [B, L].")

        if env_times.numel() != L:
            raise ValueError(
                "Length of env_times must match env_values.shape[1]."
            )

        if query_times.dim() == 2:
            if query_times.size(0) != B:
                raise ValueError(
                    "When query_times is [B, M], B must match env_values."
                )

            if not torch.allclose(
                query_times,
                query_times[0:1].expand_as(query_times),
                atol=1e-6,
            ):
                raise ValueError(
                    "query_times must be shared across batch in this implementation."
                )

            query_times = query_times[0]

        elif query_times.dim() != 1:
            raise ValueError("query_times must be [M] or [B, M].")

        return env_times, env_values, query_times

    # ------------------------------------------------------------

    # ------------------------------------------------------------

    def _interpolate_env(
        self,
        t: Tensor,
        env_times: Tensor,
        env_values: Tensor,
    ) -> Tensor:


        L = env_times.numel()

        if L < 2:
            raise ValueError("env_times must contain at least two points.")

        t = torch.clamp(t, min=env_times[0], max=env_times[-1])

        idx = torch.searchsorted(env_times, t)
        idx = idx.clamp(min=1, max=L - 1)

        t0 = env_times[idx - 1]
        t1 = env_times[idx]

        x0 = env_values[:, idx - 1, :]
        x1 = env_values[:, idx, :]

        w = ((t - t0) / (t1 - t0 + 1e-8)).view(1, 1)

        return x0 + w * (x1 - x0)

    # ------------------------------------------------------------

    # ------------------------------------------------------------

    def _rk4_step(
        self,
        h: Tensor,
        t: Tensor,
        dt: Tensor,
        env_times: Tensor,
        env_values: Tensor,
    ) -> Tensor:

        x1 = self._interpolate_env(t, env_times, env_values)
        k1 = self.dynamics(h, x1)

        x2 = self._interpolate_env(t + 0.5 * dt, env_times, env_values)
        k2 = self.dynamics(h + 0.5 * dt * k1, x2)

        x3 = self._interpolate_env(t + 0.5 * dt, env_times, env_values)
        k3 = self.dynamics(h + 0.5 * dt * k2, x3)

        x4 = self._interpolate_env(t + dt, env_times, env_values)
        k4 = self.dynamics(h + dt * k3, x4)

        return h + (dt / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )

    def _integrate_segment(
        self,
        h: Tensor,
        t0: Tensor,
        t1: Tensor,
        env_times: Tensor,
        env_values: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        dt_total = t1 - t0

        if torch.abs(dt_total).item() < 1e-10:
            return h, t1

        n_steps = max(
            1,
            int(math.ceil(torch.abs(dt_total).item() / self.cfg.max_step_h)),
        )

        dt = dt_total / n_steps
        t = t0

        for _ in range(n_steps):
            h = self._rk4_step(
                h=h,
                t=t,
                dt=dt,
                env_times=env_times,
                env_values=env_values,
            )
            t = t + dt

        return h, t1

    # ------------------------------------------------------------

    # ------------------------------------------------------------

    def _initial_state(
        self,
        env_values: Tensor,
        initial_y: Optional[Tensor],
    ) -> Tensor:

        x0 = env_values[:, 0, :]

        if initial_y is not None:
            initial_y = initial_y.to(
                device=env_values.device,
                dtype=env_values.dtype,
            )

            if initial_y.dim() != 2 or initial_y.size(-1) != self.cfg.output_dim:
                raise ValueError("initial_y must have shape [B, 3].")

            h0 = self.env_y_h0_encoder(
                torch.cat([x0, initial_y], dim=-1)
            )

        else:
            h0 = self.env_h0_encoder(x0) + self.learned_h0.view(1, -1)

        return torch.tanh(h0)

    # ------------------------------------------------------------

    # ------------------------------------------------------------

    def forward(
        self,
        env_times: Tensor,
        env_values: Tensor,
        query_times: Tensor,
        initial_y: Optional[Tensor] = None,
        return_states: bool = False,
    ) -> Dict[str, Tensor]:

        env_times, env_values, query_times = self._prepare_time_inputs(
            env_times=env_times,
            env_values=env_values,
            query_times=query_times,
        )

        if query_times.min() < env_times[0] - 1e-6:
            raise ValueError("query_times must not be earlier than env_times[0].")

        if query_times.max() > env_times[-1] + 1e-6:
            raise ValueError("query_times must not exceed env_times[-1].")

        h = self._initial_state(
            env_values=env_values,
            initial_y=initial_y,
        )

        t_current = env_times[0]

        sorted_q, sort_idx = torch.sort(query_times)

        B = env_values.size(0)
        M = sorted_q.numel()
        H = self.cfg.hidden_dim

        states_sorted = torch.empty(
            B,
            M,
            H,
            device=env_values.device,
            dtype=env_values.dtype,
        )

        for j in range(M):
            tq = sorted_q[j]

            h, t_current = self._integrate_segment(
                h=h,
                t0=t_current,
                t1=tq,
                env_times=env_times,
                env_values=env_values,
            )

            states_sorted[:, j, :] = h


        states = torch.empty_like(states_sorted)
        states[:, sort_idx, :] = states_sorted

        pred = self.readout(states)

        output = {
            "pred": pred,
            "firmness": pred[..., 0],
            "ssc": pred[..., 1],
            "ta": pred[..., 2],
        }

        if return_states:
            output["states"] = states

        return output

