from rlmpc.common.utils import read_config
from rlmpc.mpc.cartpole.acados import AcadosMPC
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from rlmpc.gym.continuous_cartpole.environment import ContinuousCartPoleSwingUpEnv  # noqa: F401

from rlmpc.common.utils import get_root_path

from stable_baselines3.common.buffers import ReplayBuffer

from typing import Any


def create_environment(config: dict) -> gym.Env:
    env = gym.make(
        config["environment"]["id"],
        render_mode=config["environment"]["render_mode"],
        min_action=-1.0,
        max_action=1.0,
        force_mag=config["environment"]["force_mag"],
    )

    return env


def plot_results(
    replay_buffer: ReplayBuffer,
    observation_labels: dict = {0: "x", 1: "v", 2: "theta", 3: "omega"},
    action_labels: dict = {0: "u"},
) -> tuple[plt.figure, Any]:
    """
    Plot replay buffer.
    """
    X = np.vstack(replay_buffer.observations[: replay_buffer.size()])
    U = np.vstack(replay_buffer.actions[: replay_buffer.size()])
    L = np.vstack(replay_buffer.rewards[: replay_buffer.size()])

    figure, axes = plt.subplots(nrows=len(observation_labels) + len(action_labels) + 1, ncols=1, sharex=True)
    for i, i_ax in enumerate(range(0, X.shape[1])):
        axes[i_ax].plot(X[:, i])
        axes[i_ax].set_ylabel(observation_labels[i])
    for i, i_ax in enumerate(range(X.shape[1], X.shape[1] + U.shape[1])):
        axes[i_ax].plot(U[:, i])
        axes[i_ax].set_ylabel(action_labels[i])
    axes[-1].plot(L)
    axes[-1].set_ylabel("cost")
    axes[-1].set_xlabel("t")

    for ax in axes:
        ax.grid()

    return figure, axes


PLOT = True


if __name__ == "__main__":
    print("Running test_acados_mpc_closed_loop.py ...")

    config = read_config(f"{get_root_path()}/config/cartpole.yaml")

    mpc = AcadosMPC(config=config["mpc"], build=True)

    x0 = np.array([0.0, 0.0, np.pi, 0.0])

    u0 = np.array([10.0])

    mpc.q_update(x0, u0)

    mpc.plot_prediction()

    # mpc.update_nlp()
