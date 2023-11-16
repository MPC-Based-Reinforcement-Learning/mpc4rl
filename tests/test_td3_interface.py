from external.stable_baselines3.stable_baselines3.td3 import TD3
from rlmpc.td3.policies import Actor, MPCTD3Policy

import gymnasium as gym
from rlmpc.common.utils import read_config

from rlmpc.gym.continuous_cartpole.environment import (
    ContinuousCartPoleBalanceEnv,
    ContinuousCartPoleSwingUpEnv,
)

from stable_baselines3.common.torch_layers import FlattenExtractor

from rlmpc.mpc.cartpole.cartpole import AcadosMPC, Config

import gymnasium as gym
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

from typing import Callable

from acados_template import AcadosOcpSolver, AcadosOcp

from stable_baselines3.common.utils import get_schedule_fn


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

    if False:
        td3_policy = MPCTD3Policy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=get_schedule_fn(1e-4),
            mpc=AcadosMPC(config=Config.from_dict(config["mpc"])),
        )

    model = TD3(
        MPCTD3Policy,
        env,
        action_noise=action_noise,
        verbose=1,
        policy_kwargs={"mpc": AcadosMPC(config=Config.from_dict(config["mpc"]))},
        train_freq=(100, "step"),
    )

    # model.learn(total_timesteps=10000, log_interval=10, progress_bar=True)

    # model.save("td3_pendulum")
    vec_env = model.get_env()

    # del model  # remove to demonstrate saving and loading

    # model = TD3.load("td3_pendulum")

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
