import os
import glob
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

from legged_lab.utils import task_registry
import legged_lab.mdp.reward_fsm as mdp_fsm  # All reward functions for FSM independence
from legged_lab.mdp.symmetry import unitree_g1_29dof
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg
from legged_lab.assets.unitree_fsm.unitree import UNITREE_G1_29DOF_CFG
from legged_lab.envs.base.base_env_config import (
    BaseAgentCfg,
    BaseEnvCfg,
    RewardCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG

@configclass
class UnitreeFSMRewardCfg(RewardCfg):
    """Rewards configuration for FSM and MAP-AMP (Multi-modal Adversarial Prior)."""
    # ========== Survival Reward ==========
    survival = RewTerm(func=mdp_fsm.survival_reward, weight=0.1)
    
    # ========== Velocity Tracking Rewards ==========
    track_lin_vel_xy_exp = RewTerm(func=mdp_fsm.track_lin_vel_xy_yaw_frame_exp, weight=3.0, params={"std": 0.5})    #1.0
    track_ang_vel_z_exp = RewTerm(func=mdp_fsm.track_ang_vel_z_world_exp, weight=3.0, params={"std": 0.5})  #-2.0
    not_moving_penalty = RewTerm(func=mdp_fsm.not_moving_penalty, weight=-1.0, params={"v_min": 0.06}) #-0.5
    not_yaw_penalty = RewTerm(func=mdp_fsm.not_yaw_penalty, weight=-1.0, params={"yaw_min": 0.05})  #-0.5
    
    # ========== Base Motion Penalties ==========
    lin_vel_z_l2 = RewTerm(func=mdp_fsm.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp_fsm.ang_vel_xy_l2, weight=-0.5)
    
    # ========== Mode-Aware Penalties (FSM-specific) ==========
    # These use reward_fsm.py functions with mode-dependent scaling
    
    energy = RewTerm(
        func=mdp_fsm.energy_mode_aware,
        weight=-1e-2,
        params={
            "stand_scale": 1.0,   # Full penalty during stand
            "walk_scale": 0.7,    # 70% penalty during walk
            "run_scale": 0.3,     # 30% penalty during run (allow high energy)
            "transition_scale": 0.5
        }
    )
    
    action_rate_l2 = RewTerm(
        func=mdp_fsm.action_rate_l2_mode_aware,
        weight=-0.1,
        params={
            "stand_scale": 1.0,
            "walk_scale": 0.7,
            "run_scale": 0.3,     # Allow rapid action changes during run
            "transition_scale": 0.5
        }
    )
    
    dof_acc_l2 = RewTerm(
        func=mdp_fsm.joint_acc_l2_mode_aware,
        weight=-2.5e-6,
        params={
            "stand_scale": 1.0,
            "walk_scale": 0.7,
            "run_scale": 0.3,     # Allow high joint acceleration for explosive movements
            "transition_scale": 0.5
        }
    )
    
    # ========== Contact and Stability Penalties ==========
    undesired_contacts = RewTerm(
        func=mdp_fsm.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle.*).*"), "threshold": 1.0},
    )
    fly = RewTerm(
        func=mdp_fsm.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 1.0},
    )
    body_orientation_l2 = RewTerm(
        func=mdp_fsm.body_orientation_l2, 
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso_link.*")}, 
        weight=-1.0 #-2.0
    )
    flat_orientation_l2 = RewTerm(func=mdp_fsm.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp_fsm.is_terminated, weight=-50.0) #-200
    
    # ========== Feet Rewards ==========
    feet_slide = RewTerm(
        func=mdp_fsm.feet_slide,
        weight=-1.0, #-0.25
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp_fsm.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_too_near = RewTerm(
        func=mdp_fsm.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "threshold": 0.15},
    )
    foot_clearance = RewTerm(
        func=mdp_fsm.foot_clearance,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "target_height": 0.1,
            "std": 0.5,
        },
    )
    
    # ========== Joint Penalties ==========
    dof_pos_limits = RewTerm(func=mdp_fsm.joint_pos_limits, weight=-1.0)
    stand_still= RewTerm(func=mdp_fsm.stand_still_joint_deviation_l1, weight=-1.0, 
                                         params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})
    
    # ========== Regularization Penalties ==========
    ankle_torque = RewTerm(func=mdp_fsm.ankle_torque, weight=-0.0005)  

@configclass
class UnitreeFSMAMPEnvCfg(BaseEnvCfg):
    """Environment configuration forFSM-AMP."""
    experiment_name: str = "unitree_fsm_amp"
    reward = UnitreeFSMRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner.prim_body_name = "pelvis"
        self.scene.robot = UNITREE_G1_29DOF_CFG
        self.scene.terrain_type = "plane"
        # self.scene.terrain_type = "generator"
        # self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
        self.robot.terminate_contacts_body_names = [
                                                        ".*torso_link.*"
                                                        # ".*knee.*",
                                                        # ".*shoulder.*",
                                                        # ".*elbow.*",
                                                    ]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*pelvis.*"]
        self.domain_rand.events.add_base_mass.params["mass_distribution_params"] = (-3.5, 3.5)
        
        # ========== Domain Randomization ==========
        self.domain_rand.events.push_robot.interval_range_s = (8, 9)
        self.commands.resampling_time_range = (6, 7)
        # ========== FSM Mode Manager Configuration ==========
        self.mode_manager.enable_mode_manager = True
        self.mode_manager.rel_stand_envs = 0.3
        self.mode_manager.rel_walk_envs = 0.3
        # rel_run_envs = 0.4 (auto-calculated as 1 - stand - walk)
        
        # Velocity ranges for each mode
        self.mode_manager.walk_vel_x_range = (-0.5, 1.5)
        self.mode_manager.walk_vel_y_range = (-0.3, 0.3)
        self.mode_manager.walk_vel_yaw_range = (-0.6, 0.6)
        
        self.mode_manager.run_vel_x_range = (1.5, 3.0)
        self.mode_manager.run_vel_y_range = (-0.3, 0.3)
        self.mode_manager.run_vel_yaw_range = (-0.6, 0.6)
        
        # Transition parameters
        self.mode_manager.transition_speed_threshold = 1.5
        self.mode_manager.speed_tolerance = 0.1
        self.mode_manager.min_mode_duration = 2.0
        self.mode_manager.smart_initialization = True

    # AMP motion files for visualization
    amp_motion_files_display = [
        "legged_lab/envs/unitree/dataset/motion_visualization/amass/run_jogging.txt"
    ]


@configclass
class UnitreeFSMAMPAgentCfg(BaseAgentCfg):
    """Agent configuration for FSM multi-gait control with MAP-AMP"""
    experiment_name: str = "unitree_fsm_amp"
    wandb_project: str = "unitree_fsm_amp"
    seed: int = 42 #42/66/88

    def __post_init__(self):
        super().__post_init__()
        
        # ========== Training Configuration ==========
        self.max_iterations = 50000
        self.save_interval = 1000
        self.resume = False
        
        # ========== MAP-AMP Component Mapping ==========
        self.algorithm.class_name = "AMPPPO"
        self.runner_class_name = "AmpOnPolicyRunner"
    amp_reward_coef = 0.4
    amp_task_reward_lerp = 0.7

    amp_motion_files = [
                            # walk
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/origin/walk_muti_walk.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/origin/walk_backwards.txt",

                            # stand
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/origin/stand_stand.txt",

                            # run
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/origin/run_jogging.txt",

                            #transition
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/origin/transition_acc.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/origin/transition_dec.txt",
    ]
    amp_num_preload_transitions = 200000
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 29 



@configclass
class UnitreeMapAMPEnvCfg(BaseEnvCfg):
    """Environment configuration for MAP-AMP (Multi-modal Adversarial Prior)."""
    experiment_name: str = "unitree_map_amp"
    reward = UnitreeFSMRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4096
        self.scene.height_scanner.prim_body_name = "pelvis"
        self.scene.robot = UNITREE_G1_29DOF_CFG
        self.scene.terrain_type = "plane"
        # self.scene.terrain_type = "generator"
        # self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
        self.robot.terminate_contacts_body_names = [
                                                    ".*torso_link.*"
                                                    # ".*knee.*",
                                                    # ".*shoulder.*",
                                                    # ".*elbow.*",
                                                ]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*pelvis.*"]
        self.domain_rand.events.add_base_mass.params["mass_distribution_params"] = (-3.5, 3.5)   #1.5
        
        # ========== Domain Randomization ==========
        self.domain_rand.events.push_robot.interval_range_s = (8, 9)
        self.commands.resampling_time_range = (6, 7)
        # ========== FSM Mode Manager Configuration ==========
        self.mode_manager.enable_mode_manager = True
        self.mode_manager.rel_stand_envs = 0.3
        self.mode_manager.rel_walk_envs = 0.3
        # rel_run_envs = 0.4 (auto-calculated as 1 - stand - walk)
        
        # Velocity ranges for each mode
        self.mode_manager.walk_vel_x_range = (-0.5, 1.5)
        self.mode_manager.walk_vel_y_range = (-0.3, 0.3)
        self.mode_manager.walk_vel_yaw_range = (-1.0, 1.0)
        
        self.mode_manager.run_vel_x_range = (1.5, 3.5)
        self.mode_manager.run_vel_y_range = (-0.3, 0.3)
        self.mode_manager.run_vel_yaw_range = (-1.0, 1.0)
        
        # Transition parameters
        self.mode_manager.transition_speed_threshold = 1.5
        self.mode_manager.speed_tolerance = 0.1
        self.mode_manager.min_mode_duration = 2.0
        self.mode_manager.smart_initialization = True

    # Run velocity curriculum: (trigger_policy_iter, new_vx_max)
        self.mode_manager.run_vx_curriculum = [
            (5000,  2.0),
            (15000, 3.0),
            (30000, 3.5),
            (40000, 4.0),
        ]
    # AMP motion files for visualization
    amp_motion_files_display = [
        "legged_lab/envs/unitree/dataset/motion_visualization/amass/run_jogging.txt"
    ]


@configclass
class UnitreeMapAMPAgentCfg(BaseAgentCfg):
    """Agent configuration for FSM multi-gait control with MAP-AMP"""
    experiment_name: str = "unitree_map_amp"
    wandb_project: str = "unitree_map_amp"
    seed: int = 42 #42/66/88

    def __post_init__(self):
        super().__post_init__()
        
        # ========== Training Configuration ==========
        self.max_iterations = 50000
        self.save_interval = 1000
        self.resume = False
        
        # ========== MAP-AMP Component Mapping ==========
        self.algorithm.class_name = "MapAMPPPO"
        self.runner_class_name = "MapAmpOnPolicyRunner"
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1,
            data_augmentation_func=unitree_g1_29dof.compute_symmetric_states
        )
    amp_reward_coef = 0.4 #0.4
    amp_task_reward_lerp = 0.7
    amp_num_disc_frames = 4
    amp_joint_pos_size = 29
    amp_joint_vel_size = 29
    amp_end_effector_pos_size = 12
    # We provide a mode mapping based on the file prefix
    mode_index_map = {"stand": 0, "walk": 1, "run": 2, "transition": 3}

    # Extend files or use existing ones (MAP-AMP resolves them properly based on filename strings)
    amp_motion_files = [
                            # walk
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/origin/walk_muti_walk.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/origin/walk_backwards.txt",

                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/walk_backwards.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/walk_backwards_1.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/walk_backwards_2.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/walk_side_step_left.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/walk_side_step_right.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/walk_turn_around.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/walk_turn_left45.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/walk_turn_left90.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/walk_turn_right45.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/walk_turn_right90.txt",

                            # stand
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/origin/stand_stand.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/stand.txt",

                            # run
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/origin/run_jogging.txt",

                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/run_change_direction.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/run_forward.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/run_forward1.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/run_turn_left45.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/run_turn_left90.txt",
                            # "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/run_turn_left135.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/run_turn_right45.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/run_turn_right90.txt",
                            # "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/run_turn_right135.txt",

                            #transition
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/origin/transition_acc.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/origin/transition_dec.txt",

                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/transition_acc0.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/transition_acc1.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/transition_acc2.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/transition_acc3.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/transition_dec0.txt",
                            "legged_lab/envs/unitree/dataset/motion_amp_expert/amass/add/transition_dec1.txt",
    ]
    amp_num_preload_transitions = 200000
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 29 
