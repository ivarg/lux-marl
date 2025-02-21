from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, flatdim


class ObsTransform:
    def __init__(self, env_params: Any):
        self.env_params = env_params
        self.n_agents = self.env_params.max_units

    def transform(self, obs: Any) -> tuple:
        obs_t = obs["player_0"]
        obs_t["steps"] = np.int64(obs_t["steps"][()])
        obs_t["match_steps"] = np.int64(obs_t["match_steps"][()])
        obs_t = tuple([obs_t] * self.n_agents)

        return obs_t

    def observation_space(self):
        num_teams = self.env_params.num_teams
        map_size = self.env_params.map_width
        max_units = self.env_params.max_units
        max_unit_energy = self.env_params.max_unit_energy
        max_energy_per_tile = self.env_params.max_energy_per_tile
        min_energy_per_tile = self.env_params.min_energy_per_tile
        max_steps_in_match = self.env_params.max_steps_in_match
        match_count_per_episode = self.env_params.match_count_per_episode
        max_relic_nodes = self.env_params.max_relic_nodes

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

        return gym.spaces.Tuple(tuple([obs] * self.n_agents))
