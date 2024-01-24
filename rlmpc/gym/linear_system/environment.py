import gymnasium as gym
import numpy as np
from typing import Optional, Union


class LinearSystemEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    def __init__(
        self,
        min_observation: np.ndarray = np.array([-0.0, -1.0], dtype=np.float32),
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
        # Penalize distance to origin

        state_cost = 0.5 * state.T @ state
        action_cost = 0.5 * action.T @ action

        lower_violation_cost = any(get_lower_violation(state, self.observation_space) > 0) * 1e1
        # lower_violation_cost = (
        #     1e2 * get_lower_violation(state, self.observation_space).T @ get_lower_violation(state, self.observation_space)
        # )

        upper_violation_cost = any(get_upper_violation(state, self.observation_space) > 0) * 1e1
        # if any(get_upper_violation(state, self.observation_space) > 0):
        #     upper_violation_cost = 1e3
        # upper_violation_cost = (
        #     1e2 * get_upper_violation(state, self.observation_space).T @ get_upper_violation(state, self.observation_space)
        # )

        # print(f"state_cost = {state_cost}")
        # print(f"action_cost = {action_cost}")
        # print(f"lower_violation_cost = {lower_violation_cost}")
        # print(f"upper_violation_cost = {upper_violation_cost}")

        return sum([state_cost, action_cost, lower_violation_cost, upper_violation_cost])

        # Penalize large actions
        # reward += action**2

        # # Penalize constraint violations
        # lower_violation = get_lower_violation(state, self.observation_space)
        # reward += 1e2 * lower_violation.T @ lower_violation

        # upper_violation = get_upper_violation(state, self.observation_space)
        # reward += 1e2 * upper_violation.T @ upper_violation

        # print(f"lower_violation_cost = {1e2*lower_violation.T @ lower_violation}")
        # print(f"upper_violation_cost = {1e2*upper_violation.T @ upper_violation}")

        # return reward


def get_lower_violation(state, observation_space):
    return np.clip(observation_space.low - state, 0, None)


def get_upper_violation(state, observation_space):
    return np.clip(state - observation_space.high, 0, None)
