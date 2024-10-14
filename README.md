# Diffusion Action Policy, in Tinygrad

This repository contains the implementation of Diffusion Action Policy, a novel approach for learning robotic control policies using diffusion models.
Original paper here: https://diffusion-policy.cs.columbia.edu/diffusion_policy_2023.pdf
## Overview
Diffusion Action Policy leverages the power of diffusion models to generate high-quality action sequences for robotic control tasks. Key features include:
* Stochastic action generation using a diffusion process
* Conditioning on current state and goal information
* Flexible architecture supporting both image and state-based inputs
* Improved robustness and consistency compared to deterministic policies

## Installation
To install the required dependencies:
```
git clone https://github.com/mdaiter/diffusion-action-policy.git
cd diffusion-action-policy
pip install diffusers tinygrad lerobot
```

## Usage

### Training
To train a new policy:
```
BEAM=2 DEBUG=2 python3.12 main_train.py
```

### Evaluation
To evaluate a new policy:
```
BEAM=2 DEBUG=2 python3.12 main_test.py
```

## Supported Tasks
* PushT
(testing others now)

## License
This project is licensed under the MIT License.
