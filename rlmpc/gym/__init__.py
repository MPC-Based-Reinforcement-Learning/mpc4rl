from gymnasium.envs.registration import register

register(
    id='ContinuousCartPoleEnv-v0',
    entry_point='rlmpc.gym.environment:ContinuousCartPoleEnv'
)