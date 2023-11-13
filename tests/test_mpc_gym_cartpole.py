"""
    Test MPC for a cartpole in gym. No learning.
"""

from typing import Optional
import gymnasium as gym
import scipy

from stable_baselines3 import PPO
from rlmpc.gym.continuous_cartpole.environment import (
    ContinuousCartPoleBalanceEnv,
    ContinuousCartPoleSwingUpEnv,
)

# from stable_baselines3.common.policies import MPC
from rlmpc.common.mpc import MPC

from rlmpc.common.utils import read_config

from stable_baselines3.common.env_util import make_vec_env

import matplotlib.pyplot as plt

from rlmpc.mpc.cartpole.cartpole import CartpoleMPC, Config


if __name__ == "__main__":
    config = read_config("config/test_mpc_gym_cartpole.yaml")

    env = gym.make(
        config["environment"]["id"],
        render_mode=config["environment"]["render_mode"],
        min_action=-1.0,
        max_action=1.0,
        force_mag=config["environment"]["force_mag"],
    )

    mpc = CartpoleMPC(config=Config.from_dict(config["mpc"]))

    model = PPO(
        "ModelPredictiveControlPolicy",
        env,
        verbose=1,
        policy_kwargs={"mpc": mpc},
    )

    # Insert training here

    vec_env = model.get_env()

    obs = vec_env.reset()

    for i in range(200):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = vec_env.step(action)

        vec_env.render("human")
