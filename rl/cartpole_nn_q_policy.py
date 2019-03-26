import gym
import numpy as np
import time
import tensorflow as tf
from scipy.misc import imresize
from random import randint
import scipy.misc

# Action space is Discrete
# Observation space is of type Box

# These terms above refer to the allowed rules
#   - Discrete space allows a fixed range of non-negative numbers
#       - In our case, (0, 1, 2, 3)
#   - Box space represents a n-dimensional box
#       - In our case, any valid observation would be an array of 210 x 160 x 3 (obs.shape)

env_name = 'CartPole-v0'
env = gym.make(env_name)
obs = env.reset()  # Resets the environment and returns the observation 

i, j = 2, 6
q = [[0 for x in range(i)] for y in range(j)]

alpha = 0.1
gamma = 0.9

actual_state = 0

IM_SIZE_1 = 142
IM_SIZE_2 = 166

def action():
    arr = q[actual_state]
    if(arr[0] == 0 and arr[1] == 0):
        _max = np.random.randint(1)
    if(arr[0] > arr[1]):
        _max = 1
    else:
        _max = 0
    #print("Estado: {} Action: {}" . format(actual_state, _max))
    return _max

def calculate_next(obs):
    pos, _, angle, _ = obs
    _sum = 0
    _sum += int((pos - (-2.4))/2.4)
    _sum += int(((angle - (-41.8))/41.8)*4)
    return _sum

max_reward = 0
max_episode = 0
max_step = 0

for i_episode in range(1000000):
    obs = env.reset()
    total_reward = 0
    avg_reward = 0
    for step in range(1000):
        act = action()
        state = actual_state
        
        obs, reward, done, info = env.step(act)  
    
        next_state = calculate_next(obs)

        actual_state = next_state
        next_action = action()

        q[state][act] = q[state][act] + (alpha * (reward + gamma*q[next_state][next_action] - q[state][act]))

        total_reward += reward
        avg_reward = total_reward/(step + 1)

        if(total_reward > max_reward):
            max_reward = total_reward
            max_episode = i_episode
            max_step = step

        env.render() 

        if done:
            print("Episode is over in {} steps. Max reward: {} in {} episode with {} steps " . format(step, max_reward, max_episode, max_step))
            break

env.close()

