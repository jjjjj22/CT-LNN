# CT-LNN

This repository contains the implementation of **CT-LNN**, a mechanism-enhanced continuous-time liquid neural network for dynamic fruit quality prediction under non-stationary cold-chain conditions.

The related manuscript is currently under review.


## Model

CT-LNN consists of three main components:

* A liquid neural network backbone for continuous-time latent quality-state modeling.
* A mechanism-guided dynamic branch to describe the influence of environmental factors on the quality evolution rate.
* An NCP residual compensation branch to capture nonlinear deviations and coupling among quality indicators.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train.py --model ct_lnn --data_dir ./data --save_dir ./checkpoints
```

Test the model:

```bash
python test.py --model ct_lnn --data_dir ./data --checkpoint ./checkpoints/best_ct_lnn.pth
```

Run prediction on a new trajectory:

```bash
python predict.py --input_file ./data/new_trajectory.xlsx --checkpoint ./checkpoints/best_ct_lnn.pth
```

