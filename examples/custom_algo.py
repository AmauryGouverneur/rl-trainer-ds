import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback


class RandomAlgo(BaseAlgorithm):
    """
    Minimal SB3-compatible algorithm stub for testing the custom loader.
    Acts randomly — useful only for verifying the loading path.
    """

    def __init__(self, policy, env: GymEnv, learning_rate=1e-3, gamma=0.99,
                 verbose=0, tensorboard_log=None, **kwargs):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            support_multi_env=False,
            supported_action_spaces=None,
        )
        self.gamma = gamma
        self._setup_model()

    def _setup_model(self):
        self.policy = None  # no neural net needed

    def learn(self, total_timesteps: int, callback: MaybeCallback = None,
              reset_num_timesteps: bool = True, progress_bar: bool = False, **kwargs):
        self._init_callback(callback)
        callback.on_training_start(locals(), globals())

        obs, _ = self.env.reset()
        self.num_timesteps = 0

        callback.on_rollout_start()
        while self.num_timesteps < total_timesteps:
            action = [self.env.action_space.sample()]
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.num_timesteps += 1
            self.locals = {"infos": [info]}
            if not callback.on_step():
                break
            if terminated or truncated:
                obs, _ = self.env.reset()
        callback.on_rollout_end()
        callback.on_training_end()
        return self
