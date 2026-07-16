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

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable
from rsl_rl.storage.map_replay_buffer import MapAMPReplayBuffer


class MapAMPPPO:
    """Proximal Policy Optimization algorithm with Multi-modal Adversarial Motion Priors."""

    policy: ActorCritic

    def __init__(
        self,
        policy,
        discriminator,
        amp_data,
        amp_normalizer,
        amp_replay_buffer_size=100000,
        min_std=None,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # Multi-frame discriminator
        num_disc_frames: int = 2,
    ):
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        if rnd_cfg is not None:
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_cfg.get("learning_rate", 1e-3))
        else:
            self.rnd = None
            self.rnd_optimizer = None

        if symmetry_cfg is not None:
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError("Data augmentation enabled but function is not callable.")
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # MAP Discriminator components
        self.amploss_coef = 1.0
        self.min_std = min_std
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_transition = RolloutStorage.Transition()
        
        # Instantiate MAP replay buffer
        self.num_disc_frames = num_disc_frames
        obs_dim = self.discriminator.discriminators[0].input_dim // num_disc_frames
        self.amp_storage = MapAMPReplayBuffer(
            self.discriminator.num_modes, obs_dim, amp_replay_buffer_size, device, num_frames=num_disc_frames
        )
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        
        # Create optimizer mapping for multi-discriminators
        params = [{"params": self.policy.parameters(), "name": "policy"}]
        for i, disc in enumerate(self.discriminator.discriminators):
            params.append({"params": disc.trunk.parameters(), "weight_decay": 10e-4, "name": f"amp_trunk_{i}"})
            params.append({"params": disc.amp_linear.parameters(), "weight_decay": 10e-2, "name": f"amp_head_{i}"})
            
        self.optimizer = optim.Adam(params, lr=learning_rate)
        
        self.storage: RolloutStorage = None  
        self.transition = RolloutStorage.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def init_storage(self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape):
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        self.storage = RolloutStorage(
            training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape, rnd_state_shape, self.device
        )

    def act(self, obs, critic_obs, amp_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, amp_obs_window):
        """amp_obs_window: (num_envs, num_disc_frames, obs_dim) – assembled by the runner."""
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if self.rnd:
            rnd_state = infos["observations"]["rnd_state"]
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            self.transition.rewards += self.intrinsic_rewards
            self.transition.rnd_state = rnd_state.clone()

        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # MAP-AMP: Extract mode_id and insert N-frame window tuple into per-mode buffer
        mode_ids = infos.get('mode_id', torch.zeros(self.storage.num_envs, device=self.device, dtype=torch.long))
        frame_tuple = tuple(amp_obs_window[:, k, :] for k in range(self.num_disc_frames))
        self.amp_storage.insert(frame_tuple, mode_ids)
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        
        num_modes = self.discriminator.num_modes
        mean_amp_loss_per_mode = [0.0] * num_modes
        mean_grad_pen_per_mode = [0.0] * num_modes
        mean_policy_pred_per_mode = [0.0] * num_modes
        mean_expert_pred_per_mode = [0.0] * num_modes

        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
            
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        amp_batch_size = (self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches)
        # DIVIDE BY NUM_MODES to keep total computational cost identical to single-expert AMP
        amp_batch_size_per_mode = max(2, amp_batch_size // self.discriminator.num_modes)

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches, 
            amp_batch_size_per_mode
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches, 
            amp_batch_size_per_mode
        )

        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            (
                obs_batch, critic_obs_batch, actions_batch, target_values_batch,
                advantages_batch, returns_batch, old_actions_log_prob_batch,
                old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, rnd_state_batch,
            ) = sample

            num_aug = 1
            original_batch_size = obs_batch.shape[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy")
                critic_obs_batch, _ = data_augmentation_func(obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic")
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch))
                        - 0.5, axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy")
                    num_aug = int(obs_batch.shape[0] / original_batch_size)
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy")
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:])
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                rnd_loss = torch.nn.MSELoss()(predicted_embedding, target_embedding)

            # --- MAP Discriminator loss calculation ---
            total_amp_loss = 0
            total_grad_pen_loss = 0
            mean_pol_d = 0
            mean_exp_d = 0
            modes_count = 0
            
            for mode_idx in range(self.discriminator.num_modes):
                policy_frames = sample_amp_policy[mode_idx]  # N-tuple or None
                expert_frames = sample_amp_expert[mode_idx]  # N-tuple or None

                # Skip if no data for this mode (e.g., buffer empty or loader missing)
                if policy_frames is None or expert_frames is None:
                    continue
                modes_count += 1

                if self.amp_normalizer is not None:
                    with torch.no_grad():
                        policy_frames = tuple(
                            self.amp_normalizer.normalize_torch(f, self.device) for f in policy_frames
                        )
                        expert_frames = tuple(
                            self.amp_normalizer.normalize_torch(f, self.device) for f in expert_frames
                        )

                policy_cat = torch.cat(list(policy_frames), dim=-1)
                expert_cat = torch.cat(list(expert_frames), dim=-1)

                disc = self.discriminator.discriminators[mode_idx]
                policy_d = disc(policy_cat)
                expert_d = disc(expert_cat)

                expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
                policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
                amp_loss_m = 0.5 * (expert_loss + policy_loss)
                grad_pen_loss_m = disc.compute_grad_pen(expert_cat, lambda_=10)

                mean_amp_loss_per_mode[mode_idx] += amp_loss_m.item() if isinstance(amp_loss_m, torch.Tensor) else amp_loss_m
                mean_grad_pen_per_mode[mode_idx] += grad_pen_loss_m.item() if isinstance(grad_pen_loss_m, torch.Tensor) else grad_pen_loss_m
                mean_policy_pred_per_mode[mode_idx] += policy_d.mean().item()
                mean_expert_pred_per_mode[mode_idx] += expert_d.mean().item()

                total_amp_loss += amp_loss_m
                total_grad_pen_loss += grad_pen_loss_m
                mean_pol_d += policy_d.mean().item()
                mean_exp_d += expert_d.mean().item()

                if self.amp_normalizer is not None:
                    for f in policy_frames:
                        self.amp_normalizer.update(f.cpu().numpy())
                    for f in expert_frames:
                        self.amp_normalizer.update(f.cpu().numpy())

            loss += self.amploss_coef * total_amp_loss + self.amploss_coef * total_grad_pen_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad() 
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            
            # Normalize logging by the number of active modes to keep scale consistent with single-expert
            if modes_count > 0:
                mean_amp_loss += (total_amp_loss.item() if isinstance(total_amp_loss, torch.Tensor) else total_amp_loss) / modes_count
                mean_grad_pen_loss += (total_grad_pen_loss.item() if isinstance(total_grad_pen_loss, torch.Tensor) else total_grad_pen_loss) / modes_count
                mean_policy_pred += (mean_pol_d / modes_count)
                mean_expert_pred += (mean_exp_d / modes_count)
                
            if mean_rnd_loss is not None: mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None: mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None: mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None: mean_symmetry_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        
        for i in range(num_modes):
            mean_amp_loss_per_mode[i] /= num_updates
            mean_grad_pen_per_mode[i] /= num_updates
            mean_policy_pred_per_mode[i] /= num_updates
            mean_expert_pred_per_mode[i] /= num_updates

        self.storage.clear()

        loss_dict = {
            "surrogate_loss": mean_surrogate_loss,
            "value_loss": mean_value_loss,
            "entropy_loss": mean_entropy,
            "amp_loss": mean_amp_loss,
            "amp_grad_pen": mean_grad_pen_loss,
            "policy_pred": mean_policy_pred,
            "expert_pred": mean_expert_pred,
        }
        
        for i in range(num_modes):
            loss_dict[f"amp_loss_m{i}"] = mean_amp_loss_per_mode[i]
            loss_dict[f"amp_grad_pen_m{i}"] = mean_grad_pen_per_mode[i]
            loss_dict[f"policy_pred_m{i}"] = mean_policy_pred_per_mode[i]
            loss_dict[f"expert_pred_m{i}"] = mean_expert_pred_per_mode[i]

        if mean_rnd_loss is not None:
            loss_dict["rnd_loss"] = mean_rnd_loss
        if mean_symmetry_loss is not None:
            loss_dict["symmetry_loss"] = mean_symmetry_loss
        return loss_dict
