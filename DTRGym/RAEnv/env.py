import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RAPlaceHolder(gym.Env):
    action_space = spaces.Discrete(8)
    def __init__(self):
        n_feature = 65
        self.observation_space = spaces.Box(
            low=np.array([0.0] * n_feature),
            high=np.array([1.] * n_feature),
            shape=(n_feature,),
            dtype=np.float32,
        )

    def seed(self, seed=None):
        return


def create_RAEnv_discrete(n_act=None):
    env = RAPlaceHolder()
    return env
