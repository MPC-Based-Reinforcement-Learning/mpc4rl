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
SAVE_MODEL = False
TOTAL_TIMESTEPS = 10_000
N_ENV = 1  # 32

if __name__ == "__main__":
    if N_ENV == 1:
        env = gym.make(ENVRIONMENT_NAME, render_mode="rgb_array")
    else:
        env = make_vec_env(ENVRIONMENT_NAME, n_envs=N_ENV)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    if SAVE_MODEL:
        model.save(f"{MODEL_NAME}_{TOTAL_TIMESTEPS}")

    vec_env = model.get_env()

    obs = vec_env.reset()

    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
