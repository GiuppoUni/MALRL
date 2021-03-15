from gym.envs.registration import register


register(
    id='AirSimEnv-v1',
    entry_point='gym_airsim.envs:AirSimEnv',
)
