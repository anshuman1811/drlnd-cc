from unityagents import UnityEnvironment
import numpy as np
import random
import torch
from collections import deque
import matplotlib.pyplot as plt
from agent import DDPGAgent
import argparse, sys
import time

# Add argument for train/test mode
parser=argparse.ArgumentParser()
parser.add_argument('mode', help='train if training else test', type=str)
args=parser.parse_args()

print ("Mode = ", args.mode)

env = UnityEnvironment(file_name="C:/Users/anshmish/Desktop/Personal/Courses/DRLND/deep-reinforcement-learning/p2_continuous-control/Reacher_Windows_x86_64/Reacher_Windows_x86_64/Reacher.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
n_agents = len(env_info.agents)
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

def trainAgent(agent, n_episodes=400, timeout=1000, score_window_size=100, solution_score=30):
    print('\nTraining agent ')
    scores = []                                                 # list containing scores from each episode
    scores_window = deque(maxlen=score_window_size)             # last 'score_window_size' scores for candidate solution

    for episode in range(n_episodes):
        start_time = time.time()
        env_info = env.reset(train_mode=True)[brain_name]       # reset the environment
        states = env_info.vector_observations                   # get the current state(s)
        ep_scores = np.zeros(n_agents)
        agent.reset()

        for t in range(timeout):
            # t_start = time.time()
            # Query agent for actions
            actions = agent.act(states)

            # t_agent_action = time.time()
            env_info = env.step(actions)[brain_name]             # send the action to the environment and get feedback
            # t_env_step = time.time()

            next_states = env_info.vector_observations           # get the next state
            rewards = env_info.rewards                           # get the reward
            dones = env_info.local_done                          # see if episode has finished

            # Move the agent a step
            # t_env_obs = time.time()
            agent.step(states, actions, rewards, next_states, dones)

            ep_scores += np.array(rewards)                       # update the score(s)
            states = next_states                                 # updates the state(s)

            # t_end = time.time()

            # print('\rStep: {:d}\tGet Action: {:.3f}\tEnv Action: {:.3f}\tAgent Step: {:.3f}\tTotal: {:.3f}'.format(t,t_agent_action-t_start, t_env_step-t_agent_action, t_end-t_env_obs, t_end-t_start), end="")

            if np.any(dones):                                    # exit loop if any of the episodes finished
                break
        
        # Cache the score(s)
        # print(ep_scores)
        scores.append(np.mean(ep_scores))
        scores_window.append(np.mean(ep_scores))

        episode_time = time.time()-start_time

        # Print episode results
        print('\rEpisode {}\ttime: {:.3f}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(episode, episode_time, np.mean(ep_scores), np.mean(scores_window)), end="")

        # Print if solution score achieved
        if np.mean(scores_window)>=solution_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\n'.format(episode-100, np.mean(scores_window)))
            torch.save(agent.actor.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic.state_dict(), 'checkpoint_critic.pth')
            break
    
    print("\nTraining Completed!")
    return scores
    
def testAgent():
    print("Testing the Agent")
    agent = DDPGAgent(state_size=state_size, action_size=action_size, n_agents=n_agents, seed=0, pretrainedWeightsFile='checkpoint_actor.pth', train = False)
    env_info = env.reset(train_mode=False)[brain_name]      # reset the environment
    states = env_info.vector_observations                   # get the current state
    score = np.zeros(n_agents)                              # initialize the score
    while True:
        actions = agent.act(states)                         # select an action
        env_info = env.step(actions)[brain_name]            # send the action to the environment
        next_states = env_info.vector_observations          # get the next state
        rewards = env_info.rewards                          # get the reward
        dones = env_info.local_done                         # see if episode has finished
        score += np.array(rewards)                          # update the score
        states = next_states                                # roll over the state to next time step
        if np.any(dones):                                   # exit loop if episode finished
            break
    print("Score: {}".format(np.mean(score)))
    return score

def plotScores(scores):
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if args.mode == 'train':
    agent = DDPGAgent(state_size=state_size, action_size=action_size, n_agents=n_agents, seed=48)
    scores = trainAgent (agent, n_episodes=1000)
    plotScores(scores)
elif args.mode == 'test':
    testAgent()
else:
	print("Invalid Mode")

