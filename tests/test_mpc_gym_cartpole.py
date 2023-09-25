"""
    Test MPC for a cartpole in gym. No learning.
"""

import gymnasium as gym

from stable_baselines3 import PPO
from rlmpc.gym.continuous_cartpole.environment import (
    ContinuousCartPoleBalanceEnv,
    ContinuousCartPoleSwingUpEnv,
)
from stable_baselines3.common.env_util import make_vec_env


# TASK = "Balance"
TASK = "SwingUp"
MODEL_NAME = f"ppo_continuous_cartpole_{TASK}"
ENVRIONMENT_NAME = f"ContinuousCartPole{TASK}Env-v0"

if __name__ == "__main__":
    env = gym.make(ENVRIONMENT_NAME, render_mode="rgb_array")

    model = PPO("ModelPredictiveControlPolicy", env, verbose=1)

    vec_env = model.get_env()

    obs = vec_env.reset()

    # for i in range(1000):
    #     action, _state = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = vec_env.step(action)
    #     vec_env.render("human")
