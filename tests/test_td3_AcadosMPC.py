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

    obs = vec_env.reset()

    # Initialize ReplayBuffer
    buffer_size = 10000  # Size of the replay buffer
    replay_buffer = ReplayBuffer(buffer_size, env.observation_space, env.action_space)

    for _ in range(100):
        action, _ = model.predict(obs)
        next_obs, reward, done, info = vec_env.step(action)
        replay_buffer.add(obs=obs, next_obs=next_obs, action=action, reward=reward, done=done, infos=info)
        obs = next_obs
        vec_env.render("human")

    print("Done with data collection.")

    obs = np.array(replay_buffer.observations)

    # Stack observations vertically in a matrix
    X = np.vstack(replay_buffer.observations)
    U = np.vstack(replay_buffer.actions)

    fig, ax = plt.subplots()
    ax.plot(X[:, 0])
    ax.plot(X[:, 1])
    ax.plot(X[:, 2])
    ax.plot(X[:, 3])
    ax.plot(U)
    plt.show()
