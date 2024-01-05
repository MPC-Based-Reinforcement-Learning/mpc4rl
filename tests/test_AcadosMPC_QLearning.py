from external.stable_baselines3.stable_baselines3.td3 import TD3
from rlmpc.td3.policies import MPCTD3Policy

import gymnasium as gym
from rlmpc.common.utils import read_config

from rlmpc.gym.continuous_cartpole.environment import (
    ContinuousCartPoleBalanceEnv,
    ContinuousCartPoleSwingUpEnv,
)

from rlmpc.mpc.cartpole.acados import AcadosMPC, Config

import gymnasium as gym
import numpy as np

from rlmpc.mpc.cartpole.common import Config, Param, ModelParams

from stable_baselines3 import TD3
from stable_baselines3.common.noise import (
    NormalActionNoise,
)
from stable_baselines3.common.buffers import ReplayBuffer

import matplotlib.pyplot as plt

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

    model.policy.actor.mpc.set_theta(np.repeat(0.9, model.policy.actor.mpc.get_theta().shape[0]))

    parameter_list = []
    parameter_list.append(model.policy.actor.mpc.get_theta().full())

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

        v = []
        dQ_dp = []
        done = 0
        status = 0

        while not done and status == 0:
            action, _ = model.predict(obs)

            # Perturb action with noise
            action += np.random.normal(0.0, 0.1, action.shape[0])
            action = np.clip(action, -1.0, 1.0)

            next_obs, reward, done, info = vec_env.step(action)

            status = model.policy.actor.mpc.status

            # if status != 0:
            #     break

            replay_buffer.add(obs=obs, next_obs=next_obs, action=action, reward=reward, done=done, infos=info)

            model.policy.actor.mpc.update_nlp()
            v.append(model.policy.actor.mpc.get_V())
            dQ_dp.append(model.policy.actor.mpc.get_dQ_dp().full().reshape(-1, 1))

            obs = next_obs

        # if status != 0:
        #     continue

        total_cost.append(np.sum(replay_buffer.rewards))

        print("Done with data collection.")
        print("Total cost:", total_cost[-1])

        print("Learning step")
        dtheta = np.repeat(0.0, model.policy.actor.mpc.get_theta().shape[0]).reshape(-1, 1)
        avg_td_error = 0.0
        mpc = model.policy.actor.mpc
        mpc.reset(replay_buffer.observations[0].reshape(-1))
        for i in range(replay_buffer.size() - 1):
            state = replay_buffer.observations[i].reshape(-1)
            action = mpc.unscale_action(replay_buffer.actions[i].reshape(-1))
            status = mpc.q_update(state, action)
            dQ_dp = mpc.get_dQ_dp().full().reshape(-1, 1)
            if status != 0:
                continue

            q = mpc.get_Q()

            next_action = mpc.unscale_action(replay_buffer.actions[i + 1].reshape(-1))
            next_state = replay_buffer.next_observations[i + 1].reshape(-1)
            status = mpc.q_update(next_state, next_action)
            if status != 0:
                continue
            next_q = mpc.get_Q()

            td_error = replay_buffer.rewards[i] + gamma * next_q - q

            dtheta += lr * td_error * dQ_dp / replay_buffer.size()

            avg_td_error += td_error / replay_buffer.size()

        # Quckfix
        dtheta[-1] = dtheta[-2]

        theta = model.policy.actor.mpc.get_theta().full()
        theta += dtheta
        model.policy.actor.mpc.set_theta(theta)

        parameter_list.append(model.policy.actor.mpc.get_theta().full().copy())

        i_episode += 1

        print("theta", theta)
        print("hallo")

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
