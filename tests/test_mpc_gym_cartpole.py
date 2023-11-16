"""
    Test MPC for a cartpole in gym. No learning.
"""

from typing import Callable
import gymnasium as gym

from stable_baselines3 import PPO
from rlmpc.gym.continuous_cartpole.environment import (
    ContinuousCartPoleBalanceEnv,
    ContinuousCartPoleSwingUpEnv,
)

from rlmpc.common.utils import read_config


from rlmpc.mpc.cartpole.cartpole import AcadosMPC, Config

from rlmpc.ppo.policies import MPCMultiInputActorCriticPolicy


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


if __name__ == "__main__":
    config = read_config("config/test_mpc_gym_cartpole.yaml")

    env = gym.make(
        config["environment"]["id"],
        render_mode=config["environment"]["render_mode"],
        min_action=-1.0,
        max_action=1.0,
        force_mag=config["environment"]["force_mag"],
    )

    model = PPO(
        policy=MPCMultiInputActorCriticPolicy,
        env=env,
        verbose=1,
        policy_kwargs={"mpc": AcadosMPC(config=Config.from_dict(config["mpc"]))},
    )

    # Insert training here

    vec_env = model.get_env()

    obs = vec_env.reset()

    for i in range(200):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = vec_env.step(action)

        vec_env.render("human")
