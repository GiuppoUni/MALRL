from gym.envs.registration import register
register(
    id='AirSimEnv-v42',
    entry_point='gym_airsim.envs:AirSimEnv',
)

register(
    id='MAGEnv-v1',
    entry_point='gym_airsim.envs:multiAGEnv',
)
