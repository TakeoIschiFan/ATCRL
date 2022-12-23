from gym.envs.registration import register

register(
    id='ATCEnvironment-v1',
    entry_point='atcrl:ATCEnvironment',
)

from omgeving import ATCEnvironment