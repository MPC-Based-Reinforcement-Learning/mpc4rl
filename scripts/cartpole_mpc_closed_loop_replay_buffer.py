from rlmpc.common.utils import read_config
from mpc.cartpole._acados import AcadosMPC
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from rlmpc.gym.continuous_cartpole.environment import ContinuousCartPoleSwingUpEnv  # noqa: F401

from rlmpc.common.utils import get_root_path

from stable_baselines3.common.buffers import ReplayBuffer

from typing import Any


def create_mpc(config: dict, build=True) -> AcadosMPC:
    mpc = AcadosMPC(config=config, build=build)

    return mpc


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

    config = read_config(f"{get_root_path()}/config/cartpole_original.yaml")

    mpc = create_mpc(config=config["mpc"], build=True)

    env = create_environment(config=config)

    replay_buffer = ReplayBuffer(
        buffer_size=1000,
        observation_space=env.observation_space,
        action_space=env.action_space,
        handle_timeout_termination=False,
    )

    nstep = 500

    obs, _ = env.reset()
    done = False
    i = 0
    while not done and i < nstep - 1:
        action = mpc.get_action(obs)

        next_obs, reward, done, _, info = env.step(action.flatten().astype(np.float32))

        obs = next_obs

        replay_buffer.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done, infos=info)

        i += 1

    if done:
        print("Swing-up successful!")
        status = 0
    else:
        print("Swing-up failed ...")
        status = 1

    if PLOT:
        plot_results(replay_buffer)

        plt.show()
