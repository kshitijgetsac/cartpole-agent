# cartpole-agent
training an agent using rl on cart pole game

# Deep Q-Learning for CartPole

This repository contains an implementation of a Deep Q-Learning (DQN) agent to solve the classic CartPole environment from [OpenAI Gym](https://gym.openai.com/). The project leverages [PyTorch](https://pytorch.org/) for building and training the neural network and demonstrates key reinforcement learning concepts such as experience replay and target network updates.

## Overview

The project includes:
- **DQN Network:** A neural network that takes the environment's state as input and outputs Q-values for each action.
- **Replay Memory:** A buffer to store past experiences, which are sampled randomly to break the correlation between consecutive experiences.
- **Epsilon-Greedy Strategy:** A method for balancing exploration (random actions) and exploitation (greedy actions) that decays epsilon over time.
- **Target Network:** A periodically updated copy of the policy network that helps stabilize training.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- [OpenAI Gym](https://gym.openai.com/)
- [PyTorch](https://pytorch.org/)
- NumPy

You can install the required Python packages using pip:

```bash
pip install gym torch numpy

## Runnning the code
python dqn_cartpole.py
