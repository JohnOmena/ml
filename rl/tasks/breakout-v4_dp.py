import gym
import numpy as np
import time
import math

from PIL import Image

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


def dist_to_tile(obs):
    max_dist = 175.32
    im = Image.fromarray(obs)
    roi_box = (8, 93, 152, 193)
    roi = im.crop(roi_box)
    bbox = roi.getbbox()
    dy = bbox[3] - bbox[1]
    dx = bbox[2] - bbox[0]
    dist = math.sqrt(dx**2 + dy**2)
    angle = math.degrees(math.acos(dx/dist))
    roi2 = roi.crop(bbox)
    xy = (bbox[0], bbox[1])
    pixel_at_xy = roi.getpixel(xy)

    side = None
    if pixel_at_xy.count(0) != 3:
        side = 'left'
    else:
        side = 'right'

    return side, dist/max_dist

steps = 1000

for step in range(steps):
    env.step(1)  # Always get the tile moving
    action = random_policy(env.action_space.n)
    # so don't fire two times
    while action == 1:
        action = random_policy(env.action_space.n)
    print('step {}, action {}'.format(step, action))
    obs, reward, done, info = env.step(action)  # Steps the env by one timestep
    env.render()  # Renders one frame of the environment
    time.sleep(1)  # Waits 1/10 second
    print(dist_to_tile(obs))
    if done:
        print("The game is over in {} steps".format(step))
        break

input('press enter\n')
env.close()

