import gymnasium as gym
import numpy as np
from typing import Optional, Union

PARAM = {
    "a": 0.5616,
    "b": 0.3126,
    "c": 48.43,
    "d": 0.507,
    "e": 55.0,
    "f": 0.1538,
    "g": 90.0,
    "h": 0.16,
    "M": 20.0,
    "C": 4.0,
    "U_A2": 6.84,
    "C_p": 0.07,
    "lam": 38.5,
    "lam_s": 36.6,
    "F_1": 10.0,
    "X_1": 5.0,
    "F_3": 50.0,
    "T_1": 40.0,
    "T_200": 25.0,
}


def erk4_step(f, x, u, p, h):
    k1 = f(x, u, p)
    k2 = f(x + h * k1 / 2, u, p)
    k3 = f(x + h * k2 / 2, u, p)
    k4 = f(x + h * k3, u, p)
    return x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def compute_f_expl(x: np.ndarray, u: np.ndarray, data: dict[float]) -> np.ndarray:
    """
    Explicit dynamics of the evaporation process.
    """
    # Unpack state and input

    X_2 = x[0]

    X_2_dot = (data["F_1"] * data["X_1"] - data["F_2"] * X_2) / data["M"]
    P_2_dot = (data["F_4"] - data["F_5"]) / data["C"]

    return np.array([X_2_dot, P_2_dot])


def compute_data(x: np.ndarray, u: np.ndarray, p: dict[float], stochastic=False) -> dict[float]:
    # Unpack state and input
    X_2, P_2 = x[0], x[1]
    P_100, F_200 = u[0], u[1]

    # Unpack parameters
    if stochastic:
        X_1 = np.random.normal(p["X_1"], 1)
        F_1 = np.random.normal(p["F_1"], 2)
        T_1 = np.random.normal(p["T_1"], 8)
        T_200 = np.random.normal(p["T_200"], 5)
    else:
        X_1 = p["X_1"]
        F_1 = p["F_1"]
        T_1 = p["T_1"]
        T_200 = p["T_200"]

    # Algebraic equations
    T_2 = p["a"] * P_2 + p["b"] * X_2 + p["c"]
    T_3 = p["d"] * P_2 + p["e"]

    T_100 = p["f"] * P_100 + p["g"]
    U_A1 = p["h"] * (F_1 + p["F_3"])

    Q_100 = U_A1 * (T_100 - T_2)
    F_100 = Q_100 / p["lam_s"]

    F_4 = (Q_100 - F_1 * p["C_p"] * (T_2 - T_1)) / p["lam"]
    Q_200 = p["U_A2"] * (T_3 - T_200) / (1 + (p["U_A2"] / (2 * p["C_p"] * F_200)))

    # Terms entering the dynamics
    F_5 = Q_200 / p["lam"]

    # Terms entering the cost
    F_2 = F_1 - F_4

    variables = {
        "T_2": T_2,
        "T_3": T_3,
        "U_A1": U_A1,
        "T_100": T_100,
        "Q_100": Q_100,
        "F_4": F_4,
        "Q_200": Q_200,
        "F_5": F_5,
        "F_2": F_2,
        "F_100": F_100,
        "F_1": F_1,
        "X_1": X_1,
        "T_1": T_1,
        "T_200": T_200,
        "M": p["M"],
        "C": p["C"],
        "F_2": F_2,
        "F_3": p["F_3"],
    }

    return variables


class EvaporationProcessEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    def __init__(
        self,
        min_action: np.ndarray = np.array([100.0, 100.0, 0.0]),
        max_action: np.ndarray = np.array([400, 400, 10.0]),
        min_observation: np.ndarray = np.array([0.0, 0.0]),
        max_observation: np.ndarray = np.array([200.0, 160.0]),
        param: dict[float] = PARAM,
        step_size: float = 1e0,
    ):
        self.step_size = step_size
        self.param = param

        self.action_space = gym.spaces.Box(low=min_action, high=max_action)

        self.observation_space = gym.spaces.Box(min_observation, max_observation)

        self.state = None

        self.data = None

    def step(self, action):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        self.data = compute_data(self.state, action, self.param, stochastic=True)

        # print("self.state:", self.state)
        self.state = erk4_step(compute_f_expl, self.state, action, self.data, self.step_size)
        # print("self.state:", self.state)

        reward = self._reward_fn(self.state, action, self.data)

        return np.array(self.state, dtype=np.float32), reward, False, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self.state = np.array([25, 49.743], dtype=np.float32)
        # self.state = np.array([40, 60.0], dtype=np.float32)

        return self.state, {}

    def _reward_fn(self, state, action, data):
        F_2 = data["F_2"]
        F_3 = data["F_3"]
        F_100 = data["F_100"]
        F_200 = action[1]

        return 10.09 * (F_2 + F_3) + 600.0 * F_100 + 0.6 * F_200 + 1.0 * action[2]

    def get(self, key):
        if key in self.data.keys():
            return self.data[key]
        elif key in self.param.keys():
            return self.param[key]
