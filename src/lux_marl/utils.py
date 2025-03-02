import gymnasium as gym
from omegaconf import DictConfig

from lux_marl import wrappers as mwrappers
from lux_marl.env import LuxAIMARLEnv

# def _make_env(
#     name,
#     time_limit,
#     clear_info,
#     observe_id,
#     standardise_rewards,
#     wrappers,
#     seed,
#     enable_video,
#     **kwargs,
# ):
#     if "smaclite" in name:
#         import smaclite  # noqa

#         env = gym.make(
#             name,
#             seed=seed,
#             render_mode="rgb_array" if enable_video else None,
#             **kwargs,
#         )
#         env = SMACliteWrapper(env)
#     else:
#         env = gym.make(
#             name, render_mode="rgb_array" if enable_video else None, **kwargs
#         )
#     if clear_info:
#         env = mwrappers.ClearInfo(env)
#     if time_limit:
#         env = gym.wrappers.TimeLimit(env, time_limit)
#     env = mwrappers.RecordEpisodeStatistics(env)
#     if observe_id:
#         env = mwrappers.ObserveID(env)
#     if standardise_rewards:
#         env = mwrappers.StandardiseReward(env)
#     if wrappers is not None:
#         for wrapper in wrappers:
#             wrapper = (
#                 getattr(mwrappers, wrapper)
#                 if hasattr(mwrappers, wrapper)
#                 else getattr(gym.wrappers, wrapper)
#             )
#             env = wrapper(env)

#     env.reset(seed=seed)
#     return env


def make_env(seed, wrappers, **env_config):
    env_config = DictConfig(env_config)
    env = LuxAIMARLEnv()
    env = mwrappers.RecordEpisodeStatistics(env)

    print(f"wrappers: {wrappers}")
    if wrappers is not None:
        for wrapper in wrappers:
            wrapper = (
                getattr(mwrappers, wrapper)
                if hasattr(mwrappers, wrapper)
                else getattr(gym.wrappers, wrapper)
            )
            env = wrapper(env)

    env.reset(seed=seed)
    return env
    # return _make_env(**env_config, enable_video=enable_video, seed=seed)
