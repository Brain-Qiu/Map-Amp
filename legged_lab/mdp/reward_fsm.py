# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Licensed under BSD-3-Clause.
#
# FSM-specific reward functions for multi-gait control

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.envs.mdp import *

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv
    from legged_lab.envs.unitree.unitree_env import UnitreeEnv

# ========== General Reward Functions (copied for FSM independence) ==========

def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv | UnitreeEnv , 
    std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv | UnitreeEnv , 
    std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def not_moving_penalty(
    env: BaseEnv | UnitreeEnv , 
    v_min: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w),
        asset.data.root_lin_vel_w[:, :3],
    )
    cmd = env.command_generator.command
    cmd_speed = torch.norm(cmd[:, :2], dim=1)
    # Use vector difference to correctly handle backward commands:
    # moving forward when cmd is backward should be fully penalized.
    velocity_error = torch.norm(cmd[:, :2] - vel_yaw[:, :2], dim=1)
    penalty = velocity_error / (cmd_speed.clamp(min=v_min) + 1e-6)
    penalty = penalty * env.is_move.float()
    return penalty


def not_yaw_penalty(
    env: BaseEnv | UnitreeEnv ,
    yaw_min: float = 0.05, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize yaw angular velocity deviation from command, proportional to command magnitude.
    Similar to not_moving_penalty but for yaw. Only active during walk/run (is_move)."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_yaw = env.command_generator.command[:, 2]
    actual_yaw = asset.data.root_ang_vel_w[:, 2]
    yaw_error = torch.abs(cmd_yaw - actual_yaw)
    penalty = yaw_error / (torch.abs(cmd_yaw).clamp(min=yaw_min) + 1e-6)
    penalty = penalty * env.is_move.float()
    penalty = penalty.clamp(max=1.0)  # Cap the penalty to prevent extreme values when cmd_yaw is small
    return penalty

def lin_vel_z_l2(env: BaseEnv | UnitreeEnv , 
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv | UnitreeEnv , 
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def undesired_contacts(env: BaseEnv | UnitreeEnv ,
                        threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv | UnitreeEnv , 
        threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Mode-aware penalty for both feet airborne simultaneously."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    both_airborne = torch.sum(is_contact, dim=-1) < 0.5
    if env.cfg.mode_manager.enable_mode_manager:
        penalty_mask = env.is_stand | env.is_walk
        if hasattr(env, 'is_transition') and env.is_transition.any():
            penalty_mask = penalty_mask | env.is_transition
        return both_airborne.float() * penalty_mask.float()
    else:
        return both_airborne.float()


def flat_orientation_l2(env: BaseEnv | UnitreeEnv , 
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(env.base_euler_xyz[:, :2]), dim=1)


def is_terminated(env: BaseEnv | UnitreeEnv ) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_slide(
    env: BaseEnv | UnitreeEnv , 
    sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv | UnitreeEnv , 
    sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv | UnitreeEnv , 
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def stand_still_joint_deviation_l1(
    env: BaseEnv | UnitreeEnv , 
    command_threshold: float = 0.1, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when in stand mode."""
    if env.cfg.mode_manager.enable_mode_manager:
        active = env.is_stand.float()
    else:
        command = env.command_generator.command
        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        total_command = linear_norm + angular_norm
        active = (total_command < command_threshold).float()
    return joint_deviation_l1(env, asset_cfg) * active


def body_orientation_l2(env: BaseEnv | UnitreeEnv , 
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(env: BaseEnv | UnitreeEnv , 
                 sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv | UnitreeEnv , 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=-1)
    return ( (threshold - distance).clamp(min=0) ) / threshold

def feet_air_time_mode_aware(
    env: BaseEnv | UnitreeEnv , 
    sensor_cfg: SceneEntityCfg,
    # Walk parameters (slower gait, shorter air time)
    walk_base: float = 0.10,
    walk_scale: float = 0.12,
    walk_max: float = 0.25,
    # Run parameters (faster gait, longer air time)
    run_base: float = 0.15,
    run_scale: float = 0.30,
    run_max: float = 0.45,
    # Flight phase bonus (double support airborne phase during running)
    run_flight_bonus: float = 0.5,
) -> torch.Tensor:
    """Mode-aware air time reward with gait-specific thresholds and flight phase bonus for running."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    cmd_speed = torch.norm(env.command_generator.command[:, :2], dim=1)  # [num_envs]
    if env.cfg.mode_manager.enable_mode_manager:
        base_threshold = torch.where(env.is_run, 
                                     torch.tensor(run_base, device=env.device), 
                                     torch.tensor(walk_base, device=env.device))
        speed_scale = torch.where(env.is_run, 
                                  torch.tensor(run_scale, device=env.device), 
                                  torch.tensor(walk_scale, device=env.device))
        max_threshold = torch.where(env.is_run, 
                                    torch.tensor(run_max, device=env.device), 
                                    torch.tensor(walk_max, device=env.device))
    else:
        base_threshold = torch.full((env.num_envs,), walk_base, device=env.device)
        speed_scale = torch.full((env.num_envs,), walk_scale, device=env.device)
        max_threshold = torch.full((env.num_envs,), walk_max, device=env.device)
    dynamic_threshold = base_threshold + speed_scale * cmd_speed
    dynamic_threshold = torch.clamp(dynamic_threshold, min=base_threshold, max=max_threshold)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    swing_air = torch.sum(air_time * (~in_contact).float(), dim=1)  # [num_envs]
    single_reward = torch.where(single_stance, swing_air, torch.zeros_like(swing_air))
    single_reward = torch.clamp(single_reward, max=dynamic_threshold)
    both_airborne = torch.sum(in_contact.int(), dim=1) == 0
    flight_air = torch.min(air_time, dim=1)[0]  # Use minimum of both feet to ensure both left ground
    flight_reward = torch.where(both_airborne, flight_air, torch.zeros_like(flight_air))
    if env.cfg.mode_manager.enable_mode_manager:
        flight_reward = flight_reward * env.is_run.float() * run_flight_bonus
    else:
        flight_reward = torch.zeros_like(flight_reward)  # No flight phase for non-FSM
    total_reward = single_reward + flight_reward
    total_reward = total_reward * env.is_move.float()
    return total_reward

def ankle_torque(env: BaseEnv | UnitreeEnv ) -> torch.Tensor:
    """Penalize large torques on the ankle joints."""
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.ankle_joint_ids]), dim=1)


def ankle_action(env: BaseEnv | UnitreeEnv ) -> torch.Tensor:
    """Penalize ankle joint actions."""
    return torch.sum(torch.abs(env.action[:, [4, 5, 10, 11]]), dim=1)



def foot_clearance(
    env: BaseEnv | UnitreeEnv ,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.1,
    std: float = 0.2,
) -> torch.Tensor:
    """Reward clear swing phase with proper foot height based on horizontal velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height_reward = torch.clamp(foot_height / target_height, 0, 1)
    root_vel_b = asset.data.root_lin_vel_b[:, :2]  # [num_envs, 2] in body frame
    root_speed = torch.norm(root_vel_b, dim=1, keepdim=True)  # [num_envs, 1]
    cmd_speed = torch.norm(env.command_generator.command[:, :2], dim=1, keepdim=True)  # [num_envs, 1]
    speed_error = torch.abs(root_speed - cmd_speed)
    velocity_reward = torch.exp(-speed_error / std)
    reward = height_reward * velocity_reward
    reward = torch.mean(reward, dim=1)
    reward = reward * env.is_move.float()
    return reward

def joint_pos_limits(env: BaseEnv | UnitreeEnv , 
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def energy_mode_aware(
    env: BaseEnv | UnitreeEnv ,
    stand_scale: float = 1.0,
    walk_scale: float = 0.7,
    run_scale: float = 0.3,
    transition_scale: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Mode-aware energy penalty: lower penalty during high-dynamic gaits"""
    asset: Articulation = env.scene[asset_cfg.name]
    base_penalty = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    
    # If FSM is disabled, use uniform scaling
    if not env.cfg.mode_manager.enable_mode_manager:
        return base_penalty
    
    # Apply mode-specific scaling
    scale = torch.ones(env.num_envs, device=env.device)
    scale = torch.where(env.is_stand, stand_scale, scale)
    scale = torch.where(env.is_walk, walk_scale, scale)
    scale = torch.where(env.is_run, run_scale, scale)
    
    # TRANSITION: use transition-specific scale
    if hasattr(env, 'is_transition') and env.is_transition.any():
        trans_to_run = env.is_transition & (env.transition_target == env.MODE_RUN)
        trans_to_stand = env.is_transition & (env.transition_target == env.MODE_STAND)
        scale = torch.where(trans_to_run, transition_scale, scale)
        scale = torch.where(trans_to_stand, transition_scale, scale)
    
    return base_penalty * scale


def action_rate_l2_mode_aware(
    env: BaseEnv | UnitreeEnv ,
    stand_scale: float = 1.0,
    walk_scale: float = 0.8,
    run_scale: float = 0.5,
    transition_scale: float = 0.6
) -> torch.Tensor:
    """Mode-aware action rate penalty: lower penalty during high-dynamic gaits"""
    base_penalty = torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )
    
    # If FSM is disabled, use uniform scaling
    if not env.cfg.mode_manager.enable_mode_manager:
        return base_penalty
    
    # Apply mode-specific scaling
    scale = torch.ones(env.num_envs, device=env.device)
    scale = torch.where(env.is_stand, stand_scale, scale)
    scale = torch.where(env.is_walk, walk_scale, scale)
    scale = torch.where(env.is_run, run_scale, scale)
    
    # TRANSITION
    if hasattr(env, 'is_transition') and env.is_transition.any():
        scale = torch.where(env.is_transition, transition_scale, scale)
    
    return base_penalty * scale


def joint_acc_l2_mode_aware(
    env: BaseEnv | UnitreeEnv ,
    stand_scale: float = 1.0,
    walk_scale: float = 0.8,
    run_scale: float = 0.5,
    transition_scale: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Mode-aware joint acceleration penalty: lower penalty during high-dynamic gaits"""
    asset: Articulation = env.scene[asset_cfg.name]
    base_penalty = torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)
    
    # If FSM is disabled, use uniform scaling
    if not env.cfg.mode_manager.enable_mode_manager:
        return base_penalty
    
    # Apply mode-specific scaling
    scale = torch.ones(env.num_envs, device=env.device)
    scale = torch.where(env.is_stand, stand_scale, scale)
    scale = torch.where(env.is_walk, walk_scale, scale)
    scale = torch.where(env.is_run, run_scale, scale)
    
    # TRANSITION
    if hasattr(env, 'is_transition') and env.is_transition.any():
        scale = torch.where(env.is_transition, transition_scale, scale)
    
    return base_penalty * scale


def joint_deviation_l1_mode_aware(
    env: BaseEnv | UnitreeEnv ,
    stand_scale: float = 1.0,
    walk_scale: float = 0.9,
    run_scale: float = 0.6,
    transition_scale: float = 0.7,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Mode-aware joint deviation penalty: lower penalty for arms during RUN"""
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    base_penalty = torch.sum(torch.abs(angle), dim=1)
    
    # If FSM is disabled, use uniform scaling
    if not env.cfg.mode_manager.enable_mode_manager:
        return base_penalty
    
    # Apply mode-specific scaling
    scale = torch.ones(env.num_envs, device=env.device)
    scale = torch.where(env.is_stand, stand_scale, scale)
    scale = torch.where(env.is_walk, walk_scale, scale)
    scale = torch.where(env.is_run, run_scale, scale)
    
    # TRANSITION
    if hasattr(env, 'is_transition') and env.is_transition.any():
        scale = torch.where(env.is_transition, transition_scale, scale)
    
    return base_penalty * scale


def survival_reward(env: BaseEnv | UnitreeEnv ) -> torch.Tensor:
    """Reward for staying alive at each timestep."""
    return torch.ones(env.num_envs, device=env.device, dtype=torch.float)

def base_height_exp(env: BaseEnv | UnitreeEnv , 
                    target_height: float = 0.79, std: float = 0.12):
    base_height = env.robot.data.root_pos_w[:, 2]
    height_error = base_height - target_height
    return torch.exp(-height_error * height_error / (std * std))


