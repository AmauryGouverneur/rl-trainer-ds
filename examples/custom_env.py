import numpy as np
import gymnasium
from gymnasium import spaces


class TrivialDiscreteEnv(gymnasium.Env):
    """
    Minimal custom env for testing the custom loader.
    Observation: single float in [0, 1].
    Action: discrete {0, 1}.
    Reward: +1 if action == 1 else 0.
    Episode ends after 50 steps.
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        obs = self.observation_space.sample()
        return obs, {}

    def step(self, action):
        self._step += 1
        obs = self.observation_space.sample()
        reward = float(action == 1)
        terminated = self._step >= 50
        return obs, reward, terminated, False, {}
