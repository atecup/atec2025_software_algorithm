import gym
from gym import Wrapper
import time

class EarlyDoneWrapper(Wrapper):
    def __init__(self, env, max_time=180):
        super().__init__(env)
        self.max_time = max_time
        self.count_lost = 0  # Initialize count_lost
        self.start_time = time.time()  # Initialize start time

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)  # take a step in the wrapped environment
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_time:
            truncated = True
        if action[0][2]==4:
            truncated=True
        if terminated or truncated:
            info['execution_time'] = elapsed_time
            print('elapsed_time:', elapsed_time)
        return obs, reward, terminated, truncated, info  # return the same results as the wrapped environment

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.start_time = time.time()
        return obs, info  # return the same results as the wrapped environment