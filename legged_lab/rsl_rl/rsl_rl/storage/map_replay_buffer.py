# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import numpy as np
import torch


class SingleReplayBuffer:
    """Fixed-size buffer to store experience tuples for a single mode."""

    def __init__(self, obs_dim, buffer_size, device, num_frames=2):
        self.num_frames = num_frames
        self.frames = [torch.zeros(buffer_size, obs_dim).to(device) for _ in range(num_frames)]
        self.buffer_size = buffer_size
        self.device = device
        self.step = 0
        self.num_samples = 0

    def insert(self, frame_tuple):
        """frame_tuple: tuple of num_frames tensors, each (batch, obs_dim)."""
        num_states = frame_tuple[0].shape[0]
        if num_states == 0:
            return
        start_idx = self.step
        end_idx = self.step + num_states
        for k, f in enumerate(frame_tuple):
            if end_idx > self.buffer_size:
                self.frames[k][self.step : self.buffer_size] = f[: self.buffer_size - self.step]
                self.frames[k][: end_idx - self.buffer_size] = f[self.buffer_size - self.step :]
            else:
                self.frames[k][start_idx:end_idx] = f
        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def sample(self, mini_batch_size):
        if self.num_samples == 0:
            return None
        sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
        return tuple(self.frames[k][sample_idxs].to(self.device) for k in range(self.num_frames))


class MapAMPReplayBuffer:
    """
    A unified buffer routing states to N distinct SingleReplayBuffers 
    based on the specific mode of the simulated trajectory.
    """

    def __init__(self, num_modes, obs_dim, buffer_size_per_mode, device, num_frames=2):
        """
        Initialize multiple replay buffers, one for each mode.
        """
        self.num_modes = num_modes
        self.num_frames = num_frames
        self.device = device
        self.buffers = [
            SingleReplayBuffer(obs_dim, buffer_size_per_mode, device, num_frames=num_frames)
            for _ in range(num_modes)
        ]

    def insert(self, frame_tuple, mode_ids):
        """
        Route and add new window frames to memory based on mode_ids.
        frame_tuple: tuple of num_frames tensors, each (num_envs, obs_dim).
        """
        for i in range(self.num_modes):
            mask = (mode_ids == i).squeeze()
            if mask.any():
                masked_tuple = tuple(f[mask] for f in frame_tuple)
                self.buffers[i].insert(masked_tuple)

    def feed_forward_generator(self, num_mini_batch, mini_batch_size_per_mode):
        """
        Yields a dict mapping mode_id to a tuple of num_frames tensors.
        If a buffer is empty, maps to None.
        """
        for _ in range(num_mini_batch):
            samples = {}
            for i in range(self.num_modes):
                samples[i] = self.buffers[i].sample(mini_batch_size_per_mode)
            yield samples
