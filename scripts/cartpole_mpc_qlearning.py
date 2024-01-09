from rlmpc.common.utils import read_config
from rlmpc.mpc.cartpole.acados import AcadosMPC
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm

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


def perturb_action(
    action: np.ndarray, noise_scale: float = 0.1, action_space: gym.spaces.Box = gym.spaces.Box(low=-1.0, high=1.0)
) -> np.ndarray:
    return np.clip(action + noise_scale * np.random.randn(*action.shape), action_space.low, action_space.high)


PLOT = True

n_episodes = 5

max_steps = 500

gamma = 0.99

lr = 1e-4


if __name__ == "__main__":
    print("Running test_acados_mpc_closed_loop.py ...")

    config = read_config(f"{get_root_path()}/config/cartpole.yaml")

    mpc = create_mpc(config=config["mpc"], build=True)

    env = create_environment(config=config)

    replay_buffer = ReplayBuffer(
        buffer_size=1000,
        observation_space=env.observation_space,
        action_space=env.action_space,
        handle_timeout_termination=False,
    )

    i_episode = 0
    while i_episode < n_episodes:
        obs, _ = env.reset()
        mpc.reset(obs)
        replay_buffer.reset()
        done = False

        i_step = 0
        while not done and i_step < max_steps - 1:
            action = mpc.get_action(obs)

            action = perturb_action(action)

            next_obs, reward, done, _, info = env.step(action.flatten().astype(np.float32))

            obs = next_obs

            replay_buffer.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done, infos=info)

            i_step += 1

        print("Total cost", np.sum(replay_buffer.rewards[: replay_buffer.size()]))

        plot_results(replay_buffer)

        plt.show()

        dtheta = np.zeros(mpc.get_theta().shape)

        mpc.reset(replay_buffer.observations[0].reshape(-1))
        for i_sample in tqdm(range(replay_buffer.size() - 1), desc="Learning"):
            obs = replay_buffer.observations[i_sample].reshape(-1)
            action = replay_buffer.actions[i_sample].reshape(-1)
            cost = replay_buffer.rewards[i_sample]
            next_obs = replay_buffer.next_observations[i_sample].reshape(-1)

            action = mpc.unscale_action(action)

            status = mpc.q_update(obs, action)
            mpc.update_nlp()
            dQ_dp = mpc.get_dQ_dp()
            q = mpc.get_Q()

            next_action = mpc.unscale_action(replay_buffer.actions[i_sample + 1].reshape(-1))
            status = mpc.q_update(next_obs, next_action)
            mpc.update_nlp()
            next_q = mpc.get_Q()

            td_error = cost + gamma * next_q - q

            dtheta += lr * td_error * dQ_dp / replay_buffer.size()

        theta = mpc.get_theta().copy()
        theta_new = theta + dtheta

        mpc.set_theta(theta_new)
        mpc.update_nlp()

        print("theta", theta)
        print("theta_new", theta_new)

        print("")

        i_episode += 1
