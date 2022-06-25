# Wrappers for MARLGrid Environments

import gym
import numpy as np
import multiprocessing as mp

class PermuteObs(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self):
        obs = self.env.reset()
        obs = [obs.transpose(2, 0, 1) for obs in obs]

        return obs

    def step(self, ac):
        obs, reward, done, info = self.env.step(ac)
        obs = [obs.transpose(2, 0, 1) for obs in obs]
        return obs, reward, done, info

class ResizeObs(gym.Wrapper):
    # Assumes square images
    def __init__(self, env, size=28):
        gym.Wrapper.__init__(self, env)
        self.size = size

    def reset(self):
        obs = self.env.reset()
        if obs[0].shape[1] != self.size:
            if obs[0].shape[1] == obs[0].shape[2]:
                obs = [np.resize(obs, (obs.shape[0], self.size, self.size)) for obs in obs]
            else:
                obs = [np.resize(obs, (self.size, self.size, obs.shape[2])) for obs in obs]

        return obs

    def step(self, ac):
        obs, reward, done, info = self.env.step(ac)
        if obs[0].shape[1] != self.size:
            if obs[0].shape[1] == obs[0].shape[2]:
                obs = [np.resize(obs, (obs.shape[0], self.size, self.size)) for obs in obs]
            else:
                obs = [np.resize(obs, (self.size, self.size, obs.shape[2])) for obs in obs]

        return obs, reward, done, info