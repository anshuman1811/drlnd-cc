# drlnd-cc
Continuous Control Project as part of the Deep Reinforcement Learning Nanodegree at Udacity

# Introduction
The project aims to solve an environment similar to the Tennis Unity environment employing Policy-Based Methods for Deep Reinforcement Learning. The objective is to train a robotic arm which can follow a continuously moving target.

# Getting Started
Follow this [link](https://github.com/udacity/deep-reinforcement-learning#dependencies) to setup the Udacity DRLND conda enviroment.

There are 2 ways to explore the code:
1. By following the guided iPython notebook - _Continuous_Control.ipynb_
2. By directly running _continuouscontrol.py_ from command line
   * Command Line argument to be passed : _train_ for training and _test_ for testing
   
       `($) python continuouscontrol.py train`
   
       `($) python continuouscontrol.py test`
   
Download the environment for this project as per your OS from the following links:
- Version 1: One (1) Agent
  - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
  - Linux Headless: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip)
  - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
  - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
  - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
- Version 2: Twenty (20) Agents
  - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
  - Linux Headless: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip)
  - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
  - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
  - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Update the path to the environment in the respecting file (_Continuous_Control.ipynb_ or _continuouscontrol.py_)    

- **Mac**: `path/to/Reacher.app`,
- **Windows** (x86): `path/to/Reacher_Windows_x86/Reacher.exe`,
- **Windows** (x86_64): `path/to/Reacher_Windows_x86_64/Reacher.exe`,
- **Linux** (x86): `path/to/Reacher_Linux/Reacher.x86`,
- **Linux** (x86_64): `path/to/Reacher_Linux/Reacher.x86_64`,
- **Linux** (x86, headless): `path/to/Reacher_Linux_NoVis/Reacher.x86`,
- **Linux** (x86_64, headless): `path/to/Reacher_Linux_NoVis/Reacher.x86_64`,

Update the following files to tweak the model or the agent:
- `model.py` defines the Neural Network
- `agent.py`defines the behavior of the DDPG Agent

# The Environment
## State Space
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction.

## Action Space
Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Rewards
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

## Solution Criteria
- Version 1: Solve the First Version
The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

- Version 2: Solve the Second Version
The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
This yields an average score for each episode (where the average is over all 20 agents).

# General Instructions
## Training
Follow the Jupyter notebook or run _continuouscontrol.py_ with argument _train_.

This will run a round of training as per details in _continuouscontrol.py_. You will be able to observe the agent performance if unity environment visualization is enabled. Upon completion, the trained model parameters will be saved in _checkpoint_actor.pth_ and _checkpoint_critic.pth_ for the _Actor_ and _Critic_ respectively.

## Testing
Follow the Jupyter notebook or run _continuouscontrol.py_ with argument _test_.

This wil run one episode of the agent with the model parameters saved in _checkpoint_actor.pth_ and _checkpoint_critic.pth_ for _Actor_ and _Critic_ respectively. You can observe the performance of the agent if the unity environment visualization is enabled.
