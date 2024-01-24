from rlmpc.mpc.linear_system.acados import AcadosMPC
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm

from rlmpc.gym.linear_system.environment import LinearSystemEnv  # noqa: F401

from rlmpc.common.utils import get_root_path


from stable_baselines3.common.buffers import ReplayBuffer


import pandas as pd


import os


def save_episode(replay_buffer: ReplayBuffer, mpc: AcadosMPC, td_error: np.ndarray, cost: np.ndarray, i_episode: int) -> None:
    state_labels = mpc.get_state_labels()
    input_labels = mpc.get_input_labels()
    state_bound_labels = [f"lb_{state_labels[i]}" for i in mpc.ocp_solver.acados_ocp.constraints.idxbx]
    state_bound_labels += [f"ub_{state_labels[i]}" for i in mpc.ocp_solver.acados_ocp.constraints.idxbx]
    input_bound_labels = [f"lb_{input_labels[i]}" for i in mpc.ocp_solver.acados_ocp.constraints.idxbu]
    input_bound_labels += [f"ub_{input_labels[i]}" for i in mpc.ocp_solver.acados_ocp.constraints.idxbu]

    episode_columns = state_labels + input_labels + state_bound_labels + input_bound_labels + ["cost", "td_error"]

    X = np.vstack([replay_buffer.observations[i].reshape(-1) for i in range(replay_buffer.size())])
    U = np.vstack([replay_buffer.actions[i].reshape(-1) for i in range(replay_buffer.size())])
    LBX = np.vstack([mpc.ocp_solver.acados_ocp.constraints.lbx for _ in range(replay_buffer.size())])
    UBX = np.vstack([mpc.ocp_solver.acados_ocp.constraints.ubx for _ in range(replay_buffer.size())])
    LBU = np.vstack([mpc.ocp_solver.acados_ocp.constraints.lbu for _ in range(replay_buffer.size())])
    UBU = np.vstack([mpc.ocp_solver.acados_ocp.constraints.ubu for _ in range(replay_buffer.size())])

    # Append nan to td_error to match size of replay buffer
    td_error = np.concatenate([td_error, np.array([np.nan] * (replay_buffer.size() - td_error.shape[0]))])
    cost = np.concatenate([cost, np.array([np.nan] * (replay_buffer.size() - cost.shape[0]))])

    data = np.hstack([X, U, LBX, UBX, LBU, UBU, cost.reshape(-1, 1), td_error.reshape(-1, 1)])

    dataframe = pd.DataFrame(index=range(replay_buffer.size()), columns=episode_columns, data=data)

    dataframe.to_csv(f"{get_res_dir()}/episode_{i_episode}.csv")


def plot_episode(replay_buffer: ReplayBuffer, mpc: AcadosMPC, td_error: np.ndarray, cost: np.ndarray) -> None:
    X = np.vstack([replay_buffer.observations[i].reshape(-1) for i in range(replay_buffer.size())])
    U = np.vstack([replay_buffer.actions[i].reshape(-1) for i in range(replay_buffer.size())])

    bounds_kwargs = {"linestyle": "-", "color": "r"}

    nrows = 5

    plt.figure(1)
    plt.subplot(nrows, 1, 1)
    plt.grid()
    plt.plot(X[:, 0])
    plt.plot(mpc.ocp_solver.acados_ocp.constraints.lbx[0] * np.ones_like(X[:, 0]), **bounds_kwargs)
    plt.plot(mpc.ocp_solver.acados_ocp.constraints.ubx[0] * np.ones_like(X[:, 0]), **bounds_kwargs)
    plt.ylabel("s_1")
    plt.subplot(nrows, 1, 2)
    plt.grid()
    plt.plot(X[:, 1])
    plt.plot(mpc.ocp_solver.acados_ocp.constraints.lbx[1] * np.ones_like(X[:, 1]), **bounds_kwargs)
    plt.plot(mpc.ocp_solver.acados_ocp.constraints.ubx[1] * np.ones_like(X[:, 1]), **bounds_kwargs)
    plt.ylabel("s_2")
    plt.subplot(nrows, 1, 3)
    plt.plot(U)
    plt.plot(mpc.ocp_solver.acados_ocp.constraints.lbu[0] * np.ones_like(U), **bounds_kwargs)
    plt.plot(mpc.ocp_solver.acados_ocp.constraints.ubu[0] * np.ones_like(U), **bounds_kwargs)
    plt.grid()
    plt.ylabel("a")
    plt.subplot(nrows, 1, 4)
    plt.plot(td_error)
    plt.grid()
    plt.ylabel("td_error")
    plt.subplot(nrows, 1, 5)
    plt.plot(cost)
    plt.ylabel("cost")
    plt.xlabel("t")
    plt.plot()
    plt.grid()
    plt.show()


INPUT_LABELS = {0: "u"}
# STATE_LABELS = {0: "x", 1: "v", 2: "theta", 3: "omega"}
# PARAM_LABELS = {0: "M", 1: "m", 2: "l", 3: "g"}

PLOT = True
N_EPISODES = 100
EPISODE_LENGTH = 100
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


if __name__ == "__main__":
    print("Running test_acados_mpc_closed_loop.py ...")

    res_dir = get_res_dir()

    param = {
        "A": np.array([[1.0, 0.25], [0.0, 1.0]]),
        "B": np.array([[0.03125], [0.25]]),
        "Q": np.identity(2),
        "R": np.identity(1),
        "b": np.array([[0.0], [0.0]]),
        "gamma": 0.9,
        "f": np.array([[0.0], [0.0], [0.0]]),
        "V_0": np.array([1e-3]),
    }

    mpc = AcadosMPC(param)

    # print(f"Cost with original parameters: {mpc.ocp_solver.get_cost()}")
    # parameter_values = mpc.get_parameter_values()
    # parameter_values[-4] = 1

    # for stage in range(mpc.ocp_solver.acados_ocp.dims.N + 1):
    #     mpc.ocp_solver.set(stage, "p", parameter_values)

    # print(f"Cost with modified parameters: {mpc.ocp_solver.get_cost()}")

    min_observation = mpc.ocp_solver.acados_ocp.constraints.lbx
    max_observation = mpc.ocp_solver.acados_ocp.constraints.ubx

    env = gym.make(
        "LinearSystemEnv-v0", lb_noise=-0.1, ub_noise=0.1, min_observation=min_observation, max_observation=max_observation
    )

    # env = create_environment(config=config)

    replay_buffer = ReplayBuffer(
        buffer_size=EPISODE_LENGTH,
        observation_space=env.observation_space,
        action_space=env.action_space,
        handle_timeout_termination=False,
    )

    parameter_values = mpc.get_parameter_values()
    parameter_labels = mpc.get_parameter_labels()
    state_labels = mpc.get_state_labels()
    input_labels = mpc.get_input_labels()
    state_bound_labels = [f"lb_{state_labels[i]}" for i in mpc.ocp_solver.acados_ocp.constraints.idxbx]
    state_bound_labels += [f"ub_{state_labels[i]}" for i in mpc.ocp_solver.acados_ocp.constraints.idxbx]
    input_bound_labels = [f"lb_{input_labels[i]}" for i in mpc.ocp_solver.acados_ocp.constraints.idxbu]
    input_bound_labels += [f"ub_{input_labels[i]}" for i in mpc.ocp_solver.acados_ocp.constraints.idxbu]

    columns = parameter_labels + ["cost", "td_error"]

    dataframe = pd.DataFrame(index=range(EPISODE_LENGTH), columns=columns)

    for i_episode in range(N_EPISODES):
        obs, _ = env.reset()
        mpc.reset(obs)
        replay_buffer.reset()
        done = False

        # Roll out episode
        for i_step in range(EPISODE_LENGTH):
            action = mpc.get_action(obs)

            next_obs, reward, done, _, info = env.step(action.flatten().astype(np.float32))

            obs = next_obs

            replay_buffer.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done, infos=info)

        total_cost = np.sum(replay_buffer.rewards[: replay_buffer.size()])

        # Learning
        n_episode_samples = replay_buffer.size() - 1
        dQ_dp = np.zeros((n_episode_samples, mpc.get_p().shape[0]))
        q = np.zeros(n_episode_samples)
        v = np.zeros(n_episode_samples)
        dp = np.zeros(mpc.get_p().shape)
        mpc.reset(replay_buffer.observations[0].reshape(-1))
        for i_sample in tqdm(
            range(n_episode_samples), desc=f"Episode {i_episode} | Total Cost: {total_cost:.2f} | Learning ..."
        ):
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

        # Collect episode data
        dataframe.loc[i_episode, parameter_labels] = mpc.get_parameter_values().flatten()
        dataframe.loc[i_episode, "cost"] = total_cost
        dataframe.loc[i_episode, "td_error"] = np.mean(td_error)

        # plot_episode(replay_buffer, mpc, td_error, cost)

        save_episode(replay_buffer, mpc, td_error, cost, i_episode)

        # Parameter update
        mpc.set_p(
            mpc.get_p() + np.mean(np.vstack([LR * td_error[i] * dQ_dp[i, :] for i in range(n_episode_samples - 1)]), axis=0)
        )

    dataframe.to_csv("data.csv")
