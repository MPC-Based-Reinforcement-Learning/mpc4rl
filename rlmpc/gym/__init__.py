from gymnasium.envs.registration import register

register(
    id="ContinuousCartPoleBalanceEnv-v0",
    entry_point="rlmpc.gym.continuous_cartpole.environment:ContinuousCartPoleBalanceEnv",
)

register(
    id="ContinuousCartPoleSwingUpEnv-v0",
    entry_point="rlmpc.gym.continuous_cartpole.environment:ContinuousCartPoleSwingUpEnv",
)
