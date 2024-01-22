import gymnasium as gym
import numpy as np
from typing import Optional, Union


class LinearSystemEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    def __init__(
        self,
        min_observation: np.ndarray = np.array([-1.0, -1.0], dtype=np.float32),
        max_observation: np.ndarray = np.array([+1.0, +1.0], dtype=np.float32),
        min_action: float = -1.0,
        max_action: float = +1.0,
        lb_noise: float = -0.1,
        ub_noise: float = +0.0,
        param: dict = {"A": np.array([[1.0, 0.25], [0.0, 1.0]]), "B": np.array([[0.03125], [0.25]])},
    ):
        self.A = param["A"]
        self.B = param["B"]
        self.lb_noise = lb_noise
        self.ub_noise = ub_noise

        self.action_space = gym.spaces.Box(low=min_action, high=max_action, shape=(1,))

        self.observation_space = gym.spaces.Box(min_observation, max_observation, dtype=np.float32)

        self.state = None

    def step(self, action):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        self.state = self.A @ self.state + self.B @ action + np.array([np.random.uniform(self.lb_noise, self.ub_noise), 0.0])

        reward = self._reward_fn(self.state, action)

        return np.array(self.state, dtype=np.float32), reward, False, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self.state = np.array([0.5, 0.5], dtype=np.float32)
        return np.array(self.state, dtype=np.float32), {}

    def _reward_fn(self, state, action):
        reward = state[0] ** 2 + state[1] ** 2
        return reward
