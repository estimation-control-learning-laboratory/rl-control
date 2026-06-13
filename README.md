# Policy-Based Reinforcement Learning for Single-Integrator Control

This repository contains a compact MATLAB implementation of a policy-gradient reinforcement-learning controller for a two-dimensional single-integrator system.

The example is intended as a beginner-friendly reinforcement-learning tutorial for students and researchers with a background in engineering, dynamics, and control systems. The code illustrates how a neural-network policy can be trained using Monte Carlo rollouts, Gaussian exploration, return normalization, and a REINFORCE-style policy-gradient update.

## Problem Description

The system is a discrete-time single integrator

[
p_{k+1} = p_k + \Delta t u_k,
]

where (p_k \in \mathbb{R}^2) is the position and (u_k \in \mathbb{R}^2) is the control input. The objective is to move the system from an initial position (p_0) to a desired goal position (p_g).

## Reinforcement-Learning Setup

The controller is represented by a small neural network policy

[
u_k = u_{\max}\tanh(W_2 \tanh(W_1 e_k + b_1) + b_2),
]

where

[
e_k = p_k - p_g.
]

During training, Gaussian exploration is added in the pre-saturation policy space. The policy is trained using a REINFORCE-style update, where the log-likelihood gradients from each episode are weighted by normalized episode returns.

## Repository Contents

```text
rl-single-integrator-control/
│
├── train_single_integrator_rl.m     # Main training and testing script
├── singleIntegratorNNPolicy.mat     # Saved trained policy, generated after training
└── README.md                        # Project description
```

## Main Features

* Two-dimensional single-integrator control example
* Small neural-network policy with one hidden layer
* Gaussian exploration during training
* Monte Carlo return evaluation
* REINFORCE-style policy-gradient update
* Advantage normalization across each training batch
* Training reward and success-rate plots
* Closed-loop trajectory tests using the learned policy

## Running the Code

Open MATLAB and run

```matlab
train_single_integrator_rl
```

The script trains the neural-network policy and saves the learned controller parameters in

```matlab
singleIntegratorNNPolicy.mat
```

After training, the script tests the learned policy on several randomly generated initial and goal positions.

## Output

The script generates three main outputs:

1. Training reward history
2. Training success-rate history
3. Closed-loop trajectories of the learned controller

The learned policy is then deployed without exploration noise.

## Requirements

This example only requires base MATLAB. No Reinforcement Learning Toolbox is needed.

## Notes

This example is intentionally simple. The goal is not to provide a state-of-the-art RL implementation, but to show the core mechanics of policy-gradient reinforcement learning in a control-oriented setting.

The example demonstrates the following RL components:

1. Policy parameterization using a neural network
2. Exploration using Gaussian perturbations
3. Trajectory generation through simulation rollouts
4. Return evaluation using discounted rewards
5. Policy optimization using REINFORCE
6. Deployment of the learned deterministic policy

## Suggested Citation

If you use this example for teaching or research, please cite the accompanying tutorial or repository.

## Author

Dr. Ankit Goel
Department of Mechanical Engineering
University of Maryland, Baltimore County
