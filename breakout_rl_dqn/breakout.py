import gymnasium as gym
import sys
import numpy as np
from pre_processing import process_input_frames
import cv2

print("Python executable:", sys.executable)


env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
observation, info = env.reset()

frames = []
for i in range(1000):
    frames.append(env.render())
    action = env.action_space.sample()  # agent policy that uses the observation and info
    if i % 4 == 0 and i != 0:
        cur_frames = frames[-4:]
        max_frame = process_input_frames(cur_frames)

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()