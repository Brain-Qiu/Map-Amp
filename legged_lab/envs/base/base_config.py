# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

import math
from dataclasses import MISSING

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.sensors.lidar import LidarCfg
from legged_lab.sensors.camera import TiledCameraCfg
@configclass
class RewardCfg:
    pass


@configclass
class HeightScannerCfg:
    enable_height_scan: bool = False
    prim_body_name: str = MISSING
    resolution: float = 0.1
    size: tuple = (1.6, 1.0)
    debug_vis: bool = False
    drift_range: tuple = (0.0, 0.0)


@configclass
class ModeManagerCfg:
    """FSM Mode Manager Configuration for multi-gait control"""
    # Enable FSM mode manager
    enable_mode_manager: bool = False
    
    # Environment distribution for intents
    rel_stand_envs: float = 0.2    # 20% environments with stand intent
    rel_walk_envs: float = 0.5     # 50% environments with walk intent
    # rel_run_envs = 1 - stand - walk = 0.3 (30% environments with run intent)
    
    # Velocity sampling ranges for different modes
    stand_vel_range: tuple = (0.0, 0.0)
    
    walk_vel_x_range: tuple = (-0.6, 1.5)
    walk_vel_y_range: tuple = (-0.5, 0.5)
    walk_vel_yaw_range: tuple = (-1.0, 1.0)
    
    run_vel_x_range: tuple = (1.5, 3.0)
    run_vel_y_range: tuple = (-0.2, 0.2)
    run_vel_yaw_range: tuple = (-0.3, 0.3)
    
    # Transition parameters
    transition_speed_threshold: float = 1.5    # Critical speed boundary (m/s)
    speed_tolerance: float = 0.1               # 10% tolerance for speed checks
    
    # Temporal control
    min_mode_duration: float = 0.5             # Minimum mode duration (seconds)
    
    # Initialization strategy
    smart_initialization: bool = True          # Use intelligent mode initialization

    # Run velocity curriculum: list of (trigger_policy_iter, new_vx_max) pairs.
    # Empty list disables curriculum. Entries are applied in order once the
    # estimated policy iteration count meets the threshold.
    run_vx_curriculum: list = []
    # num_steps_per_env from the runner config (used to estimate policy iter
    # from sim_step_counter). Default matches rsl_rl dummy_config.yaml.
    run_vx_curriculum_steps_per_iter: int = 24


@configclass
class BaseSceneCfg:
    max_episode_length_s: float = 20.0
    num_envs: int = 4096
    env_spacing: float = 2.5
    robot: ArticulationCfg = MISSING
    terrain_type: str = MISSING
    terrain_generator: TerrainGeneratorCfg = None
    max_init_terrain_level: int = 5
    height_scanner: HeightScannerCfg = HeightScannerCfg()
    lidar: LidarCfg = LidarCfg()
    depth_camera: TiledCameraCfg = TiledCameraCfg()


@configclass
class RobotCfg:
    actor_obs_history_length: int = 10
    critic_obs_history_length: int = 10
    action_scale: float = 0.25
    terminate_contacts_body_names: list = []
    feet_body_names: list = []


@configclass
class ObsScalesCfg:
    lin_vel: float = 1.0
    ang_vel: float = 1.0
    projected_gravity: float = 1.0
    commands: float = 1.0
    joint_pos: float = 1.0
    joint_vel: float = 1.0
    actions: float = 1.0
    height_scan: float = 1.0


@configclass
class NormalizationCfg:
    obs_scales: ObsScalesCfg = ObsScalesCfg()
    clip_observations: float = 100.0
    clip_actions: float = 100.0
    height_scan_offset: float = 0.5


@configclass
class CommandRangesCfg:
    lin_vel_x: tuple = (-0.6, 1.0)
    lin_vel_y: tuple = (-0.5, 0.5)
    ang_vel_z: tuple = (-1.0, 1.0)
    heading: tuple = (-math.pi, math.pi)


@configclass
class CommandsCfg:
    resampling_time_range: tuple = (10.0, 10.0)
    rel_standing_envs: float = 0.2
    rel_heading_envs: float = 1.0
    heading_command: bool = True
    heading_control_stiffness: float = 0.5
    debug_vis: bool = True
    ranges: CommandRangesCfg = CommandRangesCfg()


@configclass
class NoiseScalesCfg:
    ang_vel: float = 0.2
    projected_gravity: float = 0.05
    joint_pos: float = 0.01
    joint_vel: float = 1.5
    height_scan: float = 0.1


@configclass
class NoiseCfg:
    add_noise: bool = True
    noise_scales: NoiseScalesCfg = NoiseScalesCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                    "velocity_range": {
                        "x": (-0.3, 0.3),
                        "y": (-0.3, 0.3),
                        "z": (-0.2, 0.2),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.5, 0.5),
                    },
                },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (-0.1, 0.1),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )
    scale_joint_param = EventTerm(
                func=mdp.randomize_joint_parameters,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                    "friction_distribution_params": (0.9, 1.1),
                    "armature_distribution_params": (0.9, 1.1),
                    "operation": "scale",
                },
            ),
    scale_joint_pd = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    ),


@configclass
class ActionDelayCfg:
    enable: bool = False
    params: dict = {"max_delay": 5, "min_delay": 0}


@configclass
class DomainRandCfg:
    events: EventCfg = EventCfg()
    action_delay: ActionDelayCfg = ActionDelayCfg()


@configclass
class PhysxCfg:
    gpu_max_rigid_patch_count: int = 10 * 2**15


@configclass
class SimCfg:
    dt: float = 0.005
    decimation: int = 4
    physx: PhysxCfg = PhysxCfg()
