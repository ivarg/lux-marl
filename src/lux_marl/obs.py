from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, flatdim
from luxai_s3.params import EnvParams
from luxai_s3.wrappers import LuxAIS3GymEnv


#
class TransformLuxObs(gym.ObservationWrapper):
    def __init__(self, env: LuxAIS3GymEnv):
        super().__init__(env)
        self.n_agents = self.env.env_params.max_units
        self.observation_space = marl_observation_space(self.env.env_params)

    def observation(self, obs: dict) -> dict:
        obs_t = obs["player_0"]
        obs_t["steps"] = np.int64(obs_t["steps"][()])
        obs_t["match_steps"] = np.int64(obs_t["match_steps"][()])
        obs_t = tuple([obs_t] * self.n_agents)

        return obs_t


def marl_observation_space(env_params: EnvParams):
    num_teams = env_params.num_teams
    map_size = env_params.map_width
    max_units = env_params.max_units
    max_unit_energy = env_params.max_unit_energy
    max_energy_per_tile = env_params.max_energy_per_tile
    min_energy_per_tile = env_params.min_energy_per_tile
    max_steps_in_match = env_params.max_steps_in_match
    match_count_per_episode = env_params.match_count_per_episode
    max_relic_nodes = env_params.max_relic_nodes

    obs = Dict(
        {
            "units": Dict(
                {
                    "position": Box(
                        low=-1,
                        high=map_size - 1,
                        shape=(num_teams, max_units, 2),
                        dtype=np.int16,
                    ),
                    "energy": Box(
                        low=-1,
                        high=max_unit_energy,
                        shape=(2, max_units),
                        dtype=np.int16,
                    ),
                }
            ),
            "units_mask": Box(
                low=0, high=1, shape=(num_teams, max_units), dtype=np.bool
            ),
            "sensor_mask": Box(
                low=0,
                high=1,
                shape=(map_size, map_size),
                dtype=np.bool,
            ),
            "map_features": Dict(
                {
                    "energy": Box(
                        low=min_energy_per_tile,
                        high=max_energy_per_tile,
                        shape=(map_size, map_size),
                        dtype=np.int16,
                    ),
                    "tile_type": Box(
                        low=-1,
                        high=2,
                        shape=(map_size, map_size),
                        dtype=np.int32,
                    ),
                }
            ),
            "relic_nodes": Box(
                low=-1,
                high=map_size - 1,
                shape=(max_relic_nodes, 2),
                dtype=np.int16,
            ),
            "relic_nodes_mask": Box(
                low=0, high=1, shape=(max_relic_nodes,), dtype=np.bool
            ),
            "team_points": Box(
                low=0,
                high=np.iinfo(np.int32).max,
                shape=(num_teams,),
                dtype=np.int32,
            ),
            "team_wins": Box(
                low=0,
                high=match_count_per_episode,
                shape=(num_teams,),
                dtype=np.int32,
            ),
            "steps": Discrete((max_steps_in_match + 1) * match_count_per_episode),
            "match_steps": Discrete(max_steps_in_match + 1),
        }
    )

    return gym.spaces.Tuple(tuple([obs] * max_units))
