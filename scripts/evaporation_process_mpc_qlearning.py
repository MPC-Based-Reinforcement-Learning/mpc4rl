import numpy as np
import gymnasium as gym
import tqdm
import matplotlib.pyplot as plt
from stable_baselines3.common.buffers import ReplayBuffer
from rlmpc.gym.evaporation_process.environment import EvaporationProcessEnv, PARAM  # noqa: F401
from rlmpc.mpc.evaporation_process.acados import AcadosMPC
from scripts.evaporation_process_mpc import H_tuned, cost_param_from_H

N_EPISODES = 100
# EPISODE_LENGTH = 2000
EPISODE_LENGTH = 200
GAMMA = 0.99
LR = 1e-6


def main():
    env = gym.make("EvaporationProcessEnv-v0")

    # p_keys = ["H_lam", "h_lam", "c_lam", "H_Vf", "h_Vf", "c_Vf", "H_l", "h_l", "c_l", "c_f", "x_l", "x_u"]

    cost_param = cost_param_from_H(H_tuned)

    # cost_param["xb"]["x_l"] = np.array([25.0, 40.0])

    model_param = PARAM

    mpc = AcadosMPC(model_param=model_param, cost_param=cost_param, gamma=GAMMA)

    vars = {"F_2": [], "F_3": [], "F_100": [], "F_200": []}

    for i_episode in range(N_EPISODES):
        obs, _ = env.reset()
        replay_buffer = ReplayBuffer(EPISODE_LENGTH, env.observation_space, env.action_space, handle_timeout_termination=False)

        stage_cost = []

        for i in tqdm.tqdm(range(replay_buffer.buffer_size), desc=f"Episode {i_episode} | Collecting Samples ..."):
            action = mpc.get_action(obs)
            stage_cost.append(mpc.ocp_solver.get_cost())
            next_obs, reward, done, _, info = env.step(action.astype(np.float32))
            replay_buffer.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done, infos=info)
            obs = next_obs

        total_cost = np.sum(replay_buffer.rewards[: replay_buffer.size()])

        stage_cost = np.array(stage_cost)

        X = np.vstack([replay_buffer.observations[i].reshape(-1) for i in range(replay_buffer.size())])
        U = np.vstack([replay_buffer.actions[i].reshape(-1) for i in range(replay_buffer.size())])

        plt.figure(1)
        plt.subplot(3, 1, 1)
        plt.grid()
        plt.plot(X, label=mpc.get_state_labels())
        plt.plot(cost_param["xb"]["x_l"] * np.ones_like(X), "--", color="r", label="lower bound")
        plt.subplot(3, 1, 2)
        plt.plot(U, label=mpc.get_input_labels())
        plt.grid()
        plt.subplot(3, 1, 3)
        plt.grid()
        plt.plot(stage_cost)
        plt.ylabel("Stage Cost l(x,u)")
        plt.xlabel("Step")
        plt.legend()
        plt.show()

        # Learning
        n_episode_samples = replay_buffer.size() - 1
        dQ_dp = np.zeros((n_episode_samples, mpc.get_p().shape[0]))
        q = np.zeros(n_episode_samples)
        v = np.zeros(n_episode_samples)

        mpc.reset()

        for i_sample in tqdm.tqdm(
            range(n_episode_samples), desc=f"Episode {i_episode} | Total Cost: {total_cost:.2f} | Learning ..."
        ):
            mpc.update(replay_buffer.observations[i_sample].reshape(-1))
            v[i_sample] = mpc.get_V()

            mpc.q_update(replay_buffer.observations[i_sample].reshape(-1), replay_buffer.actions[i_sample].reshape(-1))

            dQ_dp[i_sample, :] = mpc.get_dQ_dp()
            q[i_sample] = mpc.get_Q()

        cost = replay_buffer.rewards[:n_episode_samples].reshape(-1)
        td_error = cost[:-1] + GAMMA * v[1:] - q[:-1]

        print("mpc.get_p():", mpc.get_p())

        # plt.figure(1)
        # for i in range(0, dQ_dp.shape[1]):
        #     plt.subplot(dQ_dp.shape[1] + 1, 1, i + 1)
        #     plt.grid()
        #     plt.plot(dQ_dp[:, i], label=f"dQ_dp_{i}")
        #     plt.legend()
        # plt.show()

        dp = np.mean(np.vstack([LR * td_error[i] * dQ_dp[i, :] for i in range(n_episode_samples - 1)]), axis=0)

        p_next = mpc.get_p() + dp

        # Parameter update
        mpc.set_p(mpc.get_p() + dp)

        print("dp:", dp)
        print("")

        # X = np.vstack([replay_buffer.observations[i].reshape(-1) for i in range(replay_buffer.size())])
        # U = np.vstack([replay_buffer.actions[i].reshape(-1) for i in range(replay_buffer.size())])

        # bounds_kwargs = {"linestyle": "--", "color": "r"}

        # # h = np.hstack([cost_param["xb"]["x_l"] - X, X - cost_param["xb"]["x_u"]])

        # plt.figure(1)
        # for i in range(2):
        #     plt.subplot(3, 1, i + 1)
        #     plt.grid()
        #     plt.plot(X[:, i], label=f"x_{i}")
        #     plt.plot(cost_param["xb"]["x_l"][i] * np.ones_like(X[:, i]) - Su[:, i], **bounds_kwargs, label="lower constraint")
        #     plt.legend()
        # plt.subplot(3, 1, 3)
        # plt.grid()
        # plt.plot(U, label="u")
        # plt.legend()
        # plt.show()


if __name__ == "__main__":
    main()
