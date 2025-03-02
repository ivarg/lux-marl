import math
from collections import namedtuple
from pathlib import Path

# from cpprb import ReplayBuffer
import hydra
import numpy as np
from marlbase.utils.video import VideoRecorder
from omegaconf import DictConfig

Batch = namedtuple(
    "Batch", ["obss", "actions", "rewards", "dones", "filled", "action_mask"]
)


class ReplayBuffer:
    def __init__(
        self,
        buffer_size,
        n_agents,
        observation_space,
        action_space,
        max_episode_length,
        device,
        store_action_masks=False,
    ):
        self.buffer_size = buffer_size
        self.n_agents = n_agents
        self.max_episode_length = max_episode_length
        self.store_action_masks = store_action_masks
        self.device = device

        self.pos = 0
        self.cur_pos = 0
        self.t = 0

        self.observations = [
            np.zeros(
                (max_episode_length + 1, buffer_size, *observation_space[i].shape),
                dtype=np.float32,
            )
            for i in range(n_agents)
        ]
        self.actions = np.zeros(
            (n_agents, max_episode_length, buffer_size), dtype=np.int64
        )
        self.rewards = np.zeros(
            (n_agents, max_episode_length, buffer_size), dtype=np.float32
        )
        self.dones = np.zeros((max_episode_length + 1, buffer_size), dtype=bool)
        self.filled = np.zeros((max_episode_length, buffer_size), dtype=bool)
        if store_action_masks:
            action_dim = max(action_space.n for action_space in action_space)
            self.action_masks = np.zeros(
                (n_agents, max_episode_length + 1, buffer_size, action_dim),
                dtype=np.float32,
            )

    def __len__(self):
        return min(self.pos, self.buffer_size)

    def init_episode(self, obss, action_masks=None):
        self.t = 0
        for i in range(self.n_agents):
            self.observations[i][0, self.cur_pos] = obss[i]
        if action_masks is not None:
            assert self.store_action_masks, "Action masks not stored in buffer!"
            self.action_masks[:, 0, self.cur_pos] = action_masks

    def add(self, obss, acts, rews, done, action_masks=None):
        assert self.t < self.max_episode_length, "Episode longer than given max length!"
        for i in range(self.n_agents):
            self.observations[i][self.t + 1, self.cur_pos] = obss[i]
        self.actions[:, self.t, self.cur_pos] = acts
        self.rewards[:, self.t, self.cur_pos] = rews
        self.dones[self.t + 1, self.cur_pos] = done
        self.filled[self.t, self.cur_pos] = True
        if action_masks is not None:
            assert self.store_action_masks, "Action masks not stored in buffer!"
            self.action_masks[:, self.t + 1, self.cur_pos] = action_masks
        self.t += 1

        if done:
            self.pos += 1
            self.cur_pos = self.pos % self.buffer_size
            self.t = 0

    def can_sample(self, batch_size):
        return self.pos >= batch_size

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self), size=batch_size)
        obss = torch.stack(
            [
                torch.tensor(
                    self.observations[i][:, idx],
                    dtype=torch.float32,
                    device=self.device,
                )
                for i in range(self.n_agents)
            ]
        )
        actions = torch.tensor(
            self.actions[:, :, idx], dtype=torch.int64, device=self.device
        )
        rewards = torch.tensor(
            self.rewards[:, :, idx], dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            self.dones[:, idx], dtype=torch.float32, device=self.device
        )
        filled = torch.tensor(
            self.filled[:, idx], dtype=torch.float32, device=self.device
        )
        if self.store_action_masks:
            action_mask = torch.tensor(
                self.action_masks[:, :, idx], dtype=torch.float32, device=self.device
            )
        else:
            action_mask = None
        return Batch(obss, actions, rewards, dones, filled, action_mask)
