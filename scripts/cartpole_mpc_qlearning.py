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

import pandas as pd

import os


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


def plot_episode(
    replay_buffer: ReplayBuffer, td_error: np.ndarray, dQ_dp: np.ndarray, dtheta: np.ndarray
) -> tuple[plt.figure, Any]:
    X = np.vstack(replay_buffer.observations[: replay_buffer.size()])
    U = np.vstack(replay_buffer.actions[: replay_buffer.size()])
    L = np.vstack(replay_buffer.rewards[: replay_buffer.size()])

    param_labels = PARAM_LABELS
    observation_labels = STATE_LABELS
    action_labels = INPUT_LABELS
    figure, axes = plt.subplots(
        nrows=len(param_labels) + len(observation_labels) + len(action_labels) + 3, ncols=1, sharex=True
    )
    for i, i_ax in enumerate(range(0, X.shape[1])):
        axes[i_ax].plot(X[:, i])
        axes[i_ax].set_ylabel(observation_labels[i])
    for i, i_ax in enumerate(range(X.shape[1], X.shape[1] + U.shape[1])):
        axes[i_ax].plot(U[:, i])
        axes[i_ax].set_ylabel(action_labels[i])
    for i, i_ax in enumerate(range(X.shape[1] + U.shape[1], X.shape[1] + U.shape[1] + dtheta.shape[1])):
        axes[i_ax].plot(dQ_dp[:, i])
        axes[i_ax].set_ylabel(f"dQ_d{param_labels[i]}")
    axes[-3].plot(td_error)
    axes[-3].set_ylabel("td_error")
    axes[-2].plot(dtheta)
    axes[-2].set_ylabel("dtheta")
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


INPUT_LABELS = {0: "u"}
STATE_LABELS = {0: "x", 1: "v", 2: "theta", 3: "omega"}
PARAM_LABELS = {0: "M", 1: "m", 2: "l", 3: "g"}

PLOT = True
N_EPISODES = 2
MAX_EPISODE_LENGTH = 500
GAMMA = 0.99
LR = 1e-4


def get_n_episode_samples(replay_buffer: ReplayBuffer) -> int:
    return replay_buffer.size() - 1


def get_res_dir() -> str:
    res_dir = f"{get_root_path()}/data/"

    # Append name of this script to res_dir
    res_dir += f"{os.path.basename(__file__).split('.')[0]}"

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    return res_dir


def save_episode(
    replay_buffer: ReplayBuffer,
    td_error: np.ndarray,
    dQ_dp: np.ndarray,
    dp: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    cost: np.ndarray,
    res_dir: str,
    i_episode: int,
):
    n_episode_samples = get_n_episode_samples(replay_buffer)

    columns = []
    columns += list(STATE_LABELS.values())
    columns += list(INPUT_LABELS.values())
    columns += ["q", "v", "cost", "td_error"]
    columns += [f"dQ_d{param}" for param in PARAM_LABELS.values()]
    columns += [f"d{param}" for param in PARAM_LABELS.values()]

    df = pd.DataFrame(index=range(n_episode_samples), columns=columns)

    X = np.vstack(replay_buffer.observations[:n_episode_samples])
    U = np.vstack(replay_buffer.actions[:n_episode_samples])

    for i_col, state in STATE_LABELS.items():
        df[state] = X[:, i_col]
    for i_col, input in INPUT_LABELS.items():
        df[input] = U[:, i_col]
    for i_col, param in PARAM_LABELS.items():
        df[f"dQ_d{param}"] = dQ_dp[:, i_col]
    for i_col, param in PARAM_LABELS.items():
        df[f"d{param}"] = np.concatenate((dp[:, i_col], np.array([np.nan])))
    df["q"] = q
    df["v"] = v
    df["cost"] = cost
    df["td_error"] = np.concatenate((td_error, np.array([np.nan])))

    df.to_csv(f"{res_dir}/episode_{i_episode}.csv")


if __name__ == "__main__":
    print("Running test_acados_mpc_closed_loop.py ...")

    config = read_config(f"{get_root_path()}/config/cartpole.yaml")

    res_dir = get_res_dir()

    # Ensure all model parameters are not fixed
    for param in PARAM_LABELS.values():
        config["mpc"]["model"]["params"][param]["fixed"] = False

    mpc = create_mpc(config=config["mpc"], build=True)

    env = create_environment(config=config)

    replay_buffer = ReplayBuffer(
        buffer_size=1000,
        observation_space=env.observation_space,
        action_space=env.action_space,
        handle_timeout_termination=False,
    )

    i_episode = 0
    while i_episode < N_EPISODES:
        obs, _ = env.reset()
        mpc.reset(obs)
        replay_buffer.reset()
        done = False

        i_step = 0
        while not done and i_step < MAX_EPISODE_LENGTH - 1:
            action = mpc.get_action(obs)

            action = perturb_action(action)

            next_obs, reward, done, _, info = env.step(action.flatten().astype(np.float32))

            obs = next_obs

            replay_buffer.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done, infos=info)

            i_step += 1

        print("Total cost", np.sum(replay_buffer.rewards[: replay_buffer.size()]))

        n_episode_samples = replay_buffer.size() - 1

        dQ_dp = np.zeros((n_episode_samples, mpc.get_p().shape[0]))
        q = np.zeros(n_episode_samples)
        v = np.zeros(n_episode_samples)

        dp = np.zeros(mpc.get_p().shape)

        mpc.reset(replay_buffer.observations[0].reshape(-1))
        for i_sample in tqdm(range(n_episode_samples), desc="Learning"):
            status = mpc.q_update(
                replay_buffer.observations[i_sample].reshape(-1),
                mpc.unscale_action(replay_buffer.actions[i_sample].reshape(-1)),
            )
            dQ_dp[i_sample, :] = mpc.get_dQ_dp()
            q[i_sample] = mpc.get_Q()

            mpc.update(replay_buffer.observations[i_sample].reshape(-1))
            v[i_sample] = mpc.get_V()

        cost = replay_buffer.rewards[:n_episode_samples].reshape(-1)

        td_error = cost[:-1] + GAMMA * v[1:] - q[:-1]

        dp = np.vstack([LR * td_error[i] * dQ_dp[i, :] for i in range(n_episode_samples - 1)])

        if PLOT:
            plot_episode(replay_buffer, td_error, dQ_dp, dp)
            plt.show()

        p_new = mpc.get_p() + np.mean(dp, axis=0)

        print("p", mpc.get_p())
        print("p new", p_new)

        mpc.set_p(p_new)

        # Write data to a pandas.DataFrame and store it as a csv file
        save_episode(replay_buffer, td_error, dQ_dp, dp, q, v, cost, res_dir, i_episode)

        i_episode += 1
