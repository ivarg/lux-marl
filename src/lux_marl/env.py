from typing import Any, SupportsFloat

import gymnasium as gym

# NOTE: should env_params be passed to the model?
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, flatdim
from luxai_s3.wrappers import LuxAIS3GymEnv

from .obs import TransformLuxObs


class LuxOpponent:
    def __init__(self, action_space: gym.Space):
        self.action_space = action_space

    def get_action(self, obs: dict) -> Any:
        return np.asarray(self.action_space.sample())


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
        self.env = LuxAIS3GymEnv(numpy_output=True)
        self.env = TransformLuxObs(self.env)
        self.observation_space = self.env.observation_space

        env_params = self.env.unwrapped.env_params
        self.n_agents = env_params.max_units
        self.action_space = gym.spaces.Tuple(
            tuple([gym.spaces.Discrete(5)] * self.n_agents)
        )

        self.opponent = LuxOpponent(self.action_space)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.team_points = np.zeros(2, dtype=np.int32)
        self.last_obs = obs

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
            player_1=self._make_lux_action(self.opponent.get_action(self.last_obs)),
        )
        obs, reward, terminated, truncated, info = self.env.step(action)
        # TODO: use `info` for last observation?
        self.last_obs = obs

        # obs is now a 16-length tuple - one obs for each unit/agent
        # team_points has two elements, one score for each player
        # this is a common reward game, so all agents receive the same reward
        # reward = tuple([float(obs[0]["team_points"][0])] * self.n_agents)
        reward = tuple([self._get_reward(obs[0]["team_points"])] * self.n_agents)
        terminated = np.bool(terminated["player_0"][()])
        truncated = np.bool(truncated["player_0"][()])
        return obs, reward, terminated, truncated, dict()

    def _get_reward(self, team_points: np.array) -> float:
        points = float(team_points[0])
        reward = points - float(self.team_points[0])
        self.team_points = team_points
        return reward

    def _make_lux_action(self, action: np.array) -> np.array:
        aa = np.ones((16, 3), dtype=np.int32) * -1
        aa[:, 0] = action
        return aa

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
