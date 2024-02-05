from gymnasium.envs.registration import register

register(
    id="ContinuousCartPoleBalanceEnv-v0",
    entry_point="rlmpc.gym.continuous_cartpole.environment:ContinuousCartPoleBalanceEnv",
)

register(
    id="ContinuousCartPoleSwingUpEnv-v0",
    entry_point="rlmpc.gym.continuous_cartpole.environment:ContinuousCartPoleSwingUpEnv",
)

register(
    id="LinearSystemEnv-v0",
    entry_point="rlmpc.gym.linear_system.environment:LinearSystemEnv",
)

register(
    id="EvaporationProcessEnv-v0",
    entry_point="rlmpc.gym.evaporation_process.environment:EvaporationProcessEnv",
)
