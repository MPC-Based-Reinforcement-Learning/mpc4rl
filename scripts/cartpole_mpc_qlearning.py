from rlmpc.td3.policies import MPCTD3Policy

import gymnasium as gym
from rlmpc.common.utils import read_config

from tqdm import tqdm

from rlmpc.gym.continuous_cartpole.environment import ContinuousCartPoleSwingUpEnv  # noqa: F401

from rlmpc.mpc.cartpole.acados import AcadosMPC

import numpy as np

from rlmpc.mpc.cartpole.common import Config, ModelParams

from stable_baselines3 import TD3
from stable_baselines3.common.noise import (
    NormalActionNoise,
)
from stable_baselines3.common.buffers import ReplayBuffer

import matplotlib.pyplot as plt


def plot_replay_buffer(replay_buffer: ReplayBuffer):
    fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True)

    X = np.vstack(replay_buffer.observations[: replay_buffer.size() - 1])
    U = np.vstack(replay_buffer.actions[: replay_buffer.size() - 1])

    axes[0].plot(X[:, 0])
    axes[1].plot(X[:, 1])
    axes[2].plot(X[:, 2])
    axes[3].plot(X[:, 3])

    axes[4].plot(U)

    plt.show()


if __name__ == "__main__":
    # parameter_list = []

    # p = np.repeat(0.9, 5)

    # nsteps = 10
    # for i in range(nsteps):
    #     parameter_list.append(p.copy())
    #     p += 0.01 * np.random.uniform(0.0, 1.0, p.shape[0])

    # # Create a color sheme from red to blue with nsteps
    # colors = plt.cm.RdBu(np.linspace(0, 1, nsteps))

    # fig, axes = plt.subplots()
    # for i in range(nsteps):
    #     axes.plot(parameter_list[i], linestyle="None", label=str(i), marker="o", color=colors[i])
    # plt.show()

    # config = read_config("config/test_td3_AcadosMPC.yaml")
    config = read_config("config/test_AcadosMPC.yaml")

    model_params = ModelParams.from_dict(config["mpc"]["model"]["params"])

    env = gym.make(
        config["environment"]["id"],
        render_mode=config["environment"]["render_mode"],
        min_action=-1.0,
        max_action=1.0,
        force_mag=config["environment"]["force_mag"],
    )

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3(
        MPCTD3Policy,
        env,
        action_noise=action_noise,
        verbose=1,
        policy_kwargs={"mpc": AcadosMPC(config=Config.from_dict(config["mpc"]))},
        train_freq=(100, "step"),
    )

    vec_env = model.get_env()

    # Initialize ReplayBuffer
    buffer_size = 200  # Size of the replay buffer
    replay_buffer = ReplayBuffer(buffer_size, env.observation_space, env.action_space)

    n_episodes = 100

    # model.policy.actor.mpc.set_theta(np.repeat(0.9, model.policy.actor.mpc.get_theta().shape[0]))

    p = model.policy.actor.mpc.nlp.p.val

    parameter_list = []
    parameter_list.append(model.policy.actor.mpc.get_theta())

    total_cost = []

    # Learning rate
    lr = 1e-4
    gamma = 0.99

    i_episode = 0

    # for i_episode in range(n_episodes):
    while i_episode < n_episodes:
        print("Episode", i_episode)

        # Reset the environment
        obs = vec_env.reset()
        model.policy.actor.mpc.reset(obs[0])
        replay_buffer.reset()

        print("Rollout")

        done = 0
        status = 0

        rollout_step = 0
        action_noise = True
        while not done and status == 0:
            # print("Rollout step", rollout_step)
            action, _ = model.predict(obs)

            # Perturb action with noise
            if action_noise:
                action += np.random.normal(0.0, 0.05, action.shape[0])
                action = np.clip(action, -1.0, 1.0)

            next_obs, reward, done, info = vec_env.step(action)

            # status = model.policy.actor.mpc.status

            # if status != 0:
            #     break

            replay_buffer.add(obs=obs, next_obs=next_obs, action=action, reward=reward, done=done, infos=info)

            obs = next_obs

            rollout_step += 1

        # if status != 0:
        #     continue

        total_cost.append(np.sum(replay_buffer.rewards))

        # plot_replay_buffer(replay_buffer)

        # print("Done with data collection.")
        print("Total cost:", total_cost[-1])

        # exit()

        print("Learning step")
        dtheta = np.zeros(model.policy.actor.mpc.get_theta().shape[0]).flatten()
        avg_td_error = 0.0
        mpc = model.policy.actor.mpc
        mpc.reset(replay_buffer.observations[0].reshape(-1))

        for i in tqdm(range(replay_buffer.size() - 2), desc="Training"):
            state = replay_buffer.observations[i].reshape(-1)
            action = mpc.unscale_action(replay_buffer.actions[i].reshape(-1))
            status = mpc.q_update(state, action)
            dQ_dp = mpc.get_dQ_dp()
            if status != 0:
                print("status", status)
                continue

            q = mpc.get_Q()

            next_action = mpc.unscale_action(replay_buffer.actions[i + 1].reshape(-1))
            next_state = replay_buffer.next_observations[i + 1].reshape(-1)
            status = mpc.q_update(next_state, next_action)
            if status != 0:
                print("status", status)
                continue
            next_q = mpc.get_Q()

            td_error = replay_buffer.rewards[i] + gamma * next_q - q

            dtheta += lr * td_error * dQ_dp / replay_buffer.size()

            avg_td_error += td_error / replay_buffer.size()

        theta = model.policy.actor.mpc.get_theta().copy()

        theta_old = theta.copy()

        theta += dtheta

        model.policy.actor.mpc.set_theta(theta)

        parameter_list.append(model.policy.actor.mpc.get_theta().copy())

        i_episode += 1

        print("theta_old", theta_old)
        print("theta", theta)
        print("")

    # Plot the parameters
    # colors = plt.cm.RdBu(np.linspace(0, 1, n_episodes))

    fig, axes = plt.subplots()
    for i in range(n_episodes):
        # axes.plot(parameter_list[i], linestyle="None", label=str(i), marker="o", color=colors[i])
        axes.plot(parameter_list[i], linestyle="None", label=str(i), marker="o")

    fig, axes = plt.subplots()
    axes.plot(total_cost, linestyle="None", label=str(i), marker="o")

    plt.show()

    # print("theta", theta)
    # print("dtheta", dtheta)

    # td_error = rewards[i] + gamma * model.predict(next_observations[i])[0] - model.predict(observations[i])[0]

    # Train the model on the batch
