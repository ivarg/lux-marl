from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, flatdim
from luxai_s3.wrappers import LuxAIS3GymEnv

# NOTE: should env_params be passed to the model?


#
class LuxAIMARLEnv(gym.Env):
    """
    LuxAIMARLEnv translates observations and actions between the MARL domain
    and the Lux domain. Specifically, it defines the observation space and the
    action space in accordance with what MARL algorithms require.

    The Lux environment, where 2 agents controlling up to 16 units, will be
    translated to a MARL environment with 16 cooperating agents. For this reason,
    the MARL environment needs to provide an opponent's actions, to be joined
    with the MARL action, in order to comply with the Lux environment action
    space.

    Rewards will be returned for both Lux agents as points, but in a first
    iteration the opponent will do nothing, or random actions, and it's reward
    will not affect the MARL agents. Also, the match win/lose information will
    initially be discarded.
    """

    def __init__(self, **kwargs) -> None:
        self.lux_env = LuxAIS3GymEnv(numpy_output=True)

        self.env_params = self.lux_env.env_params
        self.n_agents = self.env_params.max_units

        low = np.zeros(3)
        low[1:] = -self.env_params.unit_sap_range
        high = np.ones(3) * 6
        high[1:] = self.env_params.unit_sap_range

        self.action_space = gym.spaces.Tuple(
            tuple([gym.spaces.Discrete(5)] * self.n_agents)
        )

        self.observation_space = gym.spaces.Tuple(
            tuple([self._get_base_observation_space()] * self.n_agents)
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self.lux_env.reset(seed=seed, options=options)
        obs = self._get_obs(obs)

        return obs, dict()

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Incoming action is from a single Discrete(5) space, so we have to
        add a dimension in order to make it comply with the underlying
        environment's action space.
        """
        action = dict(
            player_0=self._make_lux_action(action),
            player_1=self._make_lux_action(self._get_opponent_action()),
        )
        obs, reward, terminated, truncated, info = self.lux_env.step(action)

        # team_points has two elements, one score for each player
        # this is a common reward game, so all agents receive the same reward
        reward = tuple([float(obs["player_0"]["team_points"][0])] * self.n_agents)
        obs = self._get_obs(obs)

        terminated = np.bool(terminated["player_0"][()])
        truncated = np.bool(truncated["player_0"][()])
        return obs, reward, terminated, truncated, dict()

    def _make_lux_action(self, action: np.array) -> np.array:
        aa = np.ones((16, 3), dtype=np.int32) * -1
        aa[:, 0] = action
        return aa

    def _get_obs(self, obs):
        obs = obs["player_0"]
        obs["steps"] = np.int64(obs["steps"][()])
        obs["match_steps"] = np.int64(obs["match_steps"][()])
        obs = tuple([obs] * self.n_agents)
        return obs

    def _get_base_observation_space(self):
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
        print(f"luxais3 obss shape: {flatdim(obs)}")
        # print(f"luxais3 obss shape: {obs.shape}")
        return obs

    def _get_opponent_action(self) -> Any:
        return np.asarray(self.action_space.sample())


class MarlObservation:
    pass


class MarlAgent:
    """
    In Marl training (the book code), the 'agent' is represented by the model, which is an implementation of an algorithm
    The training is a specialized process and only interacts with the model, and different models have different `act()` APIs.
    In a playout environment (predict), there's no training, and there needs to be an agent that interacts with the environment and uses the trained model/algorithm internally
    The MarlAgent
    """


class LuxAgent:
    """
    This class is used by the Lux runner.
    It represents one of two players in a Lux match.
    The `act()` method receives a Lux observation and returns a Lux action.
    To process the Lux observation, it transforms it to a Marl observation and calls the MarlAgent's `act()` method, which returns a Marl action.
    The Marl action is then transformed to a Lux action and is subsequently returned.

    What is the format/structure of the Lux observation?
    - The Lux observation is a dict transform of an EnvObs, which has been processed by a `to_numpy` function, which means that lists have been made into np.array(), but nothing else
    What is the format/structure of the Marl observation?
    What is the format/structure of the Lux action?
    What is the format/structure of the Marl action?

    """

    def __init__(self):
        self.marl_agent

    def act(self, step: int, obs: dict, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit.

        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
