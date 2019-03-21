import gym
import numpy as np
import time
import math

from PIL import Image


def random_policy(n):
    action = np.random.randint(0, n)
    return action


def dist_to_tile(obs):

    # Tiles at the corners
    max_dist = 175.32

    # Image object
    im = Image.fromarray(obs)

    # Region of Interest:
    #   For now the ROI is the space between 
    #   the major tile and the first layer of the "ceiling"
    roi_box = (8, 93, 152, 193)  # by trial and error 
    roi = im.crop(roi_box)

    bbox = roi.getbbox()  # Get the coordinates from the minor tile to the major tile
    dy = bbox[3] - bbox[1]
    dx = bbox[2] - bbox[0]

    # TODO: get dist from tiles' center
    # Get dist and angle
    dist = math.sqrt(dx**2 + dy**2)  # corner to corner
    angle = math.degrees(math.acos(dx/dist))

    # First pixel--at the upper left
    xy = (bbox[0], bbox[1])
    # RGB value of it 
    pixel_at_xy = roi.getpixel(xy)

    side = 'left'

    # If the value is black, then the minor tile is at the right
    if pixel_at_xy.count(0) == 3:
        side = 'right'

    # Returns the side of the minor tile and its distance in the (0, 1) interval
    return side, dist/max_dist


def main():

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

    steps = 1000

    for step in range(steps):
        # get started
        env.step(1)
        side, dist = dist_to_tile(obs)
        print(side, dist)
        action = 3
        if side == 'right':
            action = 2
        obs, reward, done, info = env.step(action)  # Steps the env by one timestep
        env.render()  # Renders one frame of the environment
        time.sleep(1)  # Waits 1/10 second
        if done:
            print("The game is over in {} steps".format(step))
            break

    input('press enter\n')
    env.close()


main()

