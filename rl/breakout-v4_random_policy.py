import gym
import numpy as np
import time

# Action space is Discrete
# Observation space is of type Box

# These terms above refer to the allowed rules
#   - Discrete space allows a fixed range of non-negative numbers
#       - In our case, (0, 1, 2, 3)
#   - Box space represents a n-dimensional box
#       - In our case, any valid observation would be an array of 210 x 160 x 3 (obs.shape)

env_name = 'Breakout-v4'
env = gym.make(env_name)
obs = env.reset()  # Resets the environment and returns the observation 

def random_policy(n):
    action = np.random.randint(0, n)
    return action

for step in range(1000):
    action = random_policy(env.action_space.n)
    obs, reward, done, info = env.step(action)  # Steps the env by one timestep
    print(info)
    env.render()  # Renders one frame of the environment
    time.sleep(0.1)  # Waits 1/10 second
    if done:
        print("The game is over in {} steps".format(step))
        break

env.close()

