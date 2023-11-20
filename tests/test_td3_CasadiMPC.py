from external.stable_baselines3.stable_baselines3.td3 import TD3
from rlmpc.td3.policies import Actor, MPCTD3Policy

import gymnasium as gym
from rlmpc.common.utils import read_config

from rlmpc.gym.continuous_cartpole.environment import (
    ContinuousCartPoleBalanceEnv,
    ContinuousCartPoleSwingUpEnv,
)

from rlmpc.mpc.cartpole.casadi import CasadiMPC, Config

import gymnasium as gym
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import (
    NormalActionNoise,
)


if __name__ == "__main__":
    config = read_config("config/test_td3_interface.yaml")

    env = gym.make(
        config["environment"]["id"],
        render_mode=config["environment"]["render_mode"],
        min_action=-1.0,
        max_action=1.0,
        force_mag=config["environment"]["force_mag"],
    )

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )

    model = TD3(
        MPCTD3Policy,
        env,
        action_noise=action_noise,
        verbose=1,
        policy_kwargs={"mpc": CasadiMPC(config=Config.from_dict(config["mpc"]))},
        train_freq=(100, "step"),
    )

    vec_env = model.get_env()

    obs = vec_env.reset()

    while True:
        action, _ = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
