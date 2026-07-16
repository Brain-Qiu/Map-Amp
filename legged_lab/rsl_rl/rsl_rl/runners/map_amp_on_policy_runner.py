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

import os
import statistics
import time
import warnings
from collections import deque

import torch

# Suppress the cuBLAS warning triggered naturally by Isaac Sim + PyTorch's autograd.grad 
warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS.*")

import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
)
from rsl_rl.utils import Normalizer, store_code_state

from rsl_rl.runners.amp_on_policy_runner import AmpOnPolicyRunner
from rsl_rl.algorithms.map_amp_ppo import MapAMPPPO
from rsl_rl.modules.map_discriminator import MapAMPDiscriminatorGroup
from rsl_rl.utils.map_motion_loader import MapAMPLoaderGroup


class MapAmpOnPolicyRunner(AmpOnPolicyRunner):
    """On-policy runner for Multi-modal Adversarial Motion Priors training."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        if self.alg_cfg["class_name"] in ["PPO", "AMPPPO", "MapAMPPPO"]:
            self.training_type = "rl"
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]

        if self.training_type == "rl":
            if "critic" in extras["observations"]:
                self.privileged_obs_type = "critic"
            else:
                self.privileged_obs_type = None
        if self.training_type == "distillation":
            if "teacher" in extras["observations"]:
                self.privileged_obs_type = "teacher"
            else:
                self.privileged_obs_type = None

        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs

        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy = policy_class(num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg).to(self.device)

        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found.")
            self.alg_cfg["rnd_cfg"]["num_states"] = rnd_state.shape[1]
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # Extract Mode Management Info for MAP AMP
        mode_index_map = train_cfg.get("mode_index_map", {"stand": 0, "walk": 1, "run": 2, "transition": 3})
        num_modes = len(mode_index_map)
        
        # Extract per-robot DOF sizes and number of discriminator frames from train_cfg
        amp_num_disc_frames = train_cfg.get("amp_num_disc_frames", 2)
        amp_joint_pos_size = train_cfg.get("amp_joint_pos_size", 20)
        amp_joint_vel_size = train_cfg.get("amp_joint_vel_size", 20)
        amp_end_effector_pos_size = train_cfg.get("amp_end_effector_pos_size", 12)

        # init map amp loader
        amp_data = MapAMPLoaderGroup(
            device,
            time_between_frames=self.env.step_dt,
            preload_transitions=True,
            num_preload_transitions=train_cfg["amp_num_preload_transitions"],
            motion_files=train_cfg["amp_motion_files"],
            mode_index_map=mode_index_map,
            joint_pos_size=amp_joint_pos_size,
            joint_vel_size=amp_joint_vel_size,
            end_effector_pos_size=amp_end_effector_pos_size,
            num_disc_frames=amp_num_disc_frames,
        )

        # Determine observation dim from the first valid inner loader
        valid_loader = next(iter(l for l in amp_data.loaders if l is not None), None)
        amp_obs_dim = valid_loader.observation_dim if valid_loader else 58 # fallback

        amp_normalizer = Normalizer(amp_obs_dim)
        discriminator = MapAMPDiscriminatorGroup(
            num_modes,
            amp_obs_dim * amp_num_disc_frames,
            train_cfg["amp_reward_coef"],
            train_cfg["amp_discr_hidden_dims"],
            device,
            train_cfg["amp_task_reward_lerp"],
        ).to(self.device)
        min_std = torch.zeros(len(train_cfg["min_normalized_std"]), device=self.device, requires_grad=False)

        # initialize algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: MapAMPPPO = alg_class(
            policy,
            discriminator,
            amp_data,
            amp_normalizer,
            device=self.device,
            min_std=min_std,
            num_disc_frames=amp_num_disc_frames,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)

        self.alg.init_storage(
            self.training_type, self.env.num_envs, self.num_steps_per_env,
            [num_obs], [num_privileged_obs], [self.env.num_actions],
        )

        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def _configure_multi_gpu(self):
        # Set primary CUDA context to avoid warning
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)
            
        if "OMPI_COMM_WORLD_SIZE" in os.environ and "OMPI_COMM_WORLD_RANK" in os.environ:
            self.is_distributed = True
            self.gpu_world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
            self.gpu_global_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
            self.multi_gpu_cfg = {"world_size": self.gpu_world_size, "global_rank": self.gpu_global_rank}
        else:
            self.is_distributed = False
            self.gpu_world_size = 1
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()
            if self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        if self.training_type == "rl":
            if self.privileged_obs_type is not None:
                privileged_obs = extras["observations"][self.privileged_obs_type]
            else:
                privileged_obs = obs
            amp_obs = self.env.get_amp_obs_for_expert_trans()
        
        obs, privileged_obs, amp_obs = obs.to(self.device), privileged_obs.to(self.device), amp_obs.to(self.device)
        obs = self.obs_normalizer(obs)
        privileged_obs = self.privileged_obs_normalizer(privileged_obs)
        self.alg.policy.train()

        # Initialize N-frame sliding window (num_envs, N, obs_dim)
        amp_num_disc_frames = self.alg.num_disc_frames
        amp_obs_window = amp_obs.unsqueeze(1).expand(-1, amp_num_disc_frames, -1).clone()
        
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # RND buffers (if used)
        erewbuffer = deque(maxlen=100)
        irewbuffer = deque(maxlen=100)

        start_iter = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_obs, amp_obs)
                    # Label AMP transitions by the mode at the start of the step.
                    # This avoids post-step auto-switch/reset contaminating the expert routing.
                    if hasattr(self.env, "current_mode"):
                        pre_step_mode_ids = self.env.current_mode.clone()
                    elif hasattr(self.env.unwrapped, "current_mode"):
                        pre_step_mode_ids = self.env.unwrapped.current_mode.clone()
                    else:
                        pre_step_mode_ids = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.long)

                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    next_amp_obs = self.env.get_amp_obs_for_expert_trans()
                    
                    obs, rewards, dones, next_amp_obs = (
                        obs.to(self.device), rewards.to(self.device), dones.to(self.device), next_amp_obs.to(self.device),
                    )
                    
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs

                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    reset_env_ids = self.env.reset_env_ids
                    terminal_amp_states = self.env.get_amp_obs_for_expert_trans()[reset_env_ids]
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    # Update sliding window: roll left and insert newest frame
                    amp_obs_window = torch.roll(amp_obs_window, -1, dims=1)
                    amp_obs_window[:, -1, :] = next_amp_obs_with_term
                    # For reset envs, fill all history with the terminal frame
                    if reset_env_ids.numel() > 0:
                        amp_obs_window[reset_env_ids] = (
                            amp_obs_window[reset_env_ids, -1:, :].expand(-1, amp_num_disc_frames, -1).clone()
                        )

                    # Keep post-step mode only for diagnostics. AMP routing uses pre-step labels.
                    if hasattr(self.env, 'current_mode'):
                        post_step_mode_ids = self.env.current_mode.clone()
                    elif hasattr(self.env.unwrapped, 'current_mode'):
                        post_step_mode_ids = self.env.unwrapped.current_mode.clone()
                    else:
                        post_step_mode_ids = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.long)

                    # Inject pre-step mode ids so reward routing and replay insertion use source-mode semantics.
                    infos['mode_id'] = pre_step_mode_ids
                    infos['post_step_mode_id'] = post_step_mode_ids

                    # Normalize each frame independently, then flatten for discriminator
                    amp_flat = amp_obs_window.clone()  # (num_envs, N, obs_dim)
                    if self.alg.amp_normalizer is not None:
                        for k in range(amp_num_disc_frames):
                            amp_flat[:, k, :] = self.alg.amp_normalizer.normalize_torch(
                                amp_flat[:, k, :], self.device
                            )
                    amp_flat = amp_flat.reshape(self.env.num_envs, -1)  # (num_envs, N*obs_dim)

                    rewards = self.alg.discriminator.predict_amp_reward(
                        amp_flat, rewards, pre_step_mode_ids
                    )[0]
                    amp_obs = torch.clone(next_amp_obs)
                    self.alg.process_env_step(rewards, dones, infos, amp_obs_window)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                
                self.alg.compute_returns(privileged_obs)

            loss_dict = self.alg.update()
            
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Store mode specific metrics separately before self.log prints them
                mode_metrics = {k: v for k, v in loss_dict.items() if "_m" in k}
                for k in mode_metrics.keys():
                    del loss_dict[k]
                    
                # Log to tensorboard manually for the mode metrics
                if hasattr(self, 'writer') and self.writer is not None:
                    for k, v in mode_metrics.items():
                        self.writer.add_scalar(f"Loss/{k}", v, it)

                # Log information
                self.log(locals())
                
                # Restore them back just in case they are needed elsewhere
                loss_dict.update(mode_metrics)

                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def get_inference_policy(self, device=None):
        self.alg.policy.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
            self.obs_normalizer.to(device)
        return self.alg.policy.act_inference

    def get_inference_policy_with_normalizer(self, device=None):
        policy = self.get_inference_policy(device)
        return lambda obs: policy(self.obs_normalizer(obs))

    def get_amp_obs_normalizer(self, device=None):
        if device is not None:
            self.alg.amp_normalizer.to(device)
        return lambda obs: self.alg.amp_normalizer.normalize_torch(obs, device)
