import numpy as np
import gymnasium
from gymnasium import spaces


class ActionClipWrapper(gymnasium.Wrapper):
    """Clip continuous actions to the action space bounds."""
    def step(self, action):
        if isinstance(self.action_space, spaces.Box):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)


def make_env(env_id: str, shaping_config: dict):
    env = gymnasium.make(env_id)

    if shaping_config.get("obs_norm"):
        env = gymnasium.wrappers.NormalizeObservation(env)

    if shaping_config.get("frame_stack"):
        n = int(shaping_config.get("frame_stack_n", 4))
        env = gymnasium.wrappers.FrameStack(env, n)

    if shaping_config.get("act_clip"):
        env = ActionClipWrapper(env)

    if shaping_config.get("rew_clip"):
        env = gymnasium.wrappers.TransformReward(env, lambda r: np.clip(r, -1.0, 1.0))

    if shaping_config.get("rew_scale"):
        scale = float(shaping_config.get("rew_scale_v", 1.0))
        env = gymnasium.wrappers.TransformReward(env, lambda r: r * scale)

    return env
