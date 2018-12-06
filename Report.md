# Learning Algorithm

The algorithm used for the implementation was a **Deep Deterministic Policy Gradients** based on the [DeepMind paper](https://arxiv.org/pdf/1509.02971.pdf).

## Hyperparameters
### Agent Hyperparameters
```
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate for actor
LR_CRITIC = 1e-3        # learning rate for critic
WEIGHT_DECAY = 0        # L2 weight decay for critic
UPDATE_EVERY = 20
NUM_UPDATES = 10

Ornstein-Uhlenbeck Noise was used
THETA = 0.15
SIGMA = 0.2
MU = 0.0
```

### Training Hyperparameters
```
N_EPISODES=1000         # number of training episodes
TIMEOUT=1000            # limit on max number of steps in each episode
```

# Model Architecture
In contrast to the DDPG paper implementation of the agent, the model architecture of the neural network used here is as follows (defined in _model.py_)
```
Actor
FC1                 : 256 output units
FC2                 : 128 output units
FC3 (output layer)  : 4 ouput units

The outputs of FC1 and FC2 were fed to ReLU activation units.
```
```
Critic
FC1                 : 256 output units
FC2                 : 128 output units
FC3 (output layer)  : 1 ouput unit

The outputs of FC1 and FC2 were fed to ReLU activation units.
Actions were added only in FC2
```

# Weigth Initialization
In accordance with the DDPG paper, 
The final layer weights and biases of both the actor and critic were initialized from a uniform distribution [−3 × 10−3, 3 × 10−3] and [3 × 10−4, 3 × 10−4] for the low dimensional and pixel cases respectively. This was to ensure the initial outputs for the policy and value estimates were near zero. 

The other layers were initialized from uniform distributions[− 1/√f, 1/√f] where f is the fan-in of the layer.

# Training Results
The solution criteria used for training was an average score of 30.0 over 100 consecutive episodes.

The agent took about 250 episodes to solve the environment.

The average training scores of the agent:
![Average Training Scores of the Agent](/LearningCurve.png)

# Ideas for Future Work
## Prioritized Experience Replay
The idea is of prioritized replay is to increase the replay probability of experience tuples that have a high expected learning progress. It uses the absolute TD error as proxy to evaluate the same. This helps with faster learning as well as a better final policy. Reference : [Prioritized Experience Replay Paper](https://arxiv.org/abs/1511.05952)
