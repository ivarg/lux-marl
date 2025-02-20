from gymnasium import register

from lux_marl.env import LuxAIMARLEnv

register(id="LuxAIMARLEnv", entry_point="lux_marl:LuxAIMARLEnv")

