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

import torch
import torch.nn as nn
from torch import autograd


class MapAMPDiscriminator(nn.Module):
    """
    Discriminator neural network for adversarial motion priors (AMP) reward prediction.
    Used as an inner class for MapAMPDiscriminatorGroup.
    """

    def __init__(self, input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp=0.0):
        super().__init__()

        self.device = device
        self.input_dim = input_dim
        self.amp_reward_coef = amp_reward_coef
        
        amp_layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
            
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

        self.task_reward_lerp = task_reward_lerp

    def forward(self, x):
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self, expert_cat, lambda_=10):
        """Gradient penalty on pre-concatenated expert frames tensor."""
        expert_data = expert_cat
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(self, state_next_state_cat, task_reward):
        with torch.no_grad():
            self.eval()
            d = self.amp_linear(self.trunk(state_next_state_cat))
            reward = self.amp_reward_coef * torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            if self.task_reward_lerp > 0 and task_reward is not None:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
            self.train()
        return reward.squeeze(), d

    def _lerp_reward(self, disc_r, task_r):
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r


class MapAMPDiscriminatorGroup(nn.Module):
    """
    A container routing states to one of N distinct MapAMPDiscriminator networks
    based on the specific mode a simulated trajectory belongs to.
    """
    
    def __init__(self, num_modes, input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp=0.0):
        super().__init__()
        self.num_modes = num_modes
        self.device = device
        
        self.discriminators = nn.ModuleList([
            MapAMPDiscriminator(input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp)
            for _ in range(num_modes)
        ])

    def forward(self, x, mode_ids):
        """
        Forward pass for prediction routing to valid models based on their mode.
        x: (batch, input_dim)
        mode_ids: (batch,) integer tensor of mode types 
        """
        out = torch.zeros(x.shape[0], 1, device=self.device)
        for i, disc in enumerate(self.discriminators):
            mask = (mode_ids == i)
            if mask.any():
                out[mask] = disc(x[mask])
        return out

    def predict_amp_reward(self, state_cat, task_reward, mode_ids, normalizer=None):
        """
        Computes AMP rewards across discriminators.
        state_cat: pre-concatenated N-frame window, shape (num_envs, N*obs_dim_per_frame).
                   Normalization is applied per-frame by the caller before passing here.
        """
        if normalizer is not None:
            state_cat = normalizer.normalize_torch(state_cat, self.device)

        rewards = torch.zeros(state_cat.shape[0], device=self.device)
        logits = torch.zeros(state_cat.shape[0], 1, device=self.device)
        
        for i, disc in enumerate(self.discriminators):
            mask = (mode_ids == i)
            if mask.any():
                tr = task_reward[mask] if task_reward is not None else None
                r, d = disc.predict_amp_reward(state_cat[mask], tr)
                rewards[mask] = r
                logits[mask] = d
                
        return rewards, logits
