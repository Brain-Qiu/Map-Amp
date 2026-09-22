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

import argparse
import os
import yaml

import torch
from isaaclab.app import AppLauncher
from legged_lab.rsl_rl.rsl_rl.runners import OnPolicyRunner, AmpOnPolicyRunner, MapAmpOnPolicyRunner

from legged_lab.utils import task_registry

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg
from datetime import datetime
def policy_exported_param(env):
    num_actions = len(env.dof_names)
    # FSM observation layout: [command(3), mode_onehot(4), dof_pos(29), dof_vel(29), action(29), ang_vel(3), euler(3)]
    obs_index_commands = 0
    obs_index_mode_onehot = 3  # FSM mode one-hot encoding
    obs_index_q = 7  # 3 (command) + 4 (mode_onehot)
    obs_index_qd = obs_index_q + num_actions
    obs_index_action = obs_index_q + 2 * num_actions
    obs_index_imu_waist_angvel = obs_index_q + 3 * num_actions
    obs_index_imu_waist_orientation = obs_index_q + 3 * num_actions + 3
    obs_index_torque = -1
    obs_index_forcesensor = -1
    obs_index_imu_waist_linearacc = -1
    obs_index_imu_torso_angvel = -1
    obs_index_imu_torso_orientation = -1
    obs_index_imu_torso_linearacc = -1
    default_joint_pos = []
    dof_names = []
    for i in range(env.default_joint_pos.shape[1]):
        default_joint_pos.append(env.default_joint_pos[0, i].item())
    for name in env.dof_names:
        if name == "leg_left_hip_yaw_joint":
            dof_names.append("joint_L_hipY")
        if name == "leg_left_hip_roll_joint":
            dof_names.append("joint_L_hipR")
        if name == "leg_left_hip_pitch_joint":
            dof_names.append("joint_L_hipP")
        if name == "leg_left_knee_joint":
            dof_names.append("joint_L_knee")
        if name == "leg_left_ankle_roll_joint":
            dof_names.append("joint_L_ankleR")
        if name == "leg_left_ankle_pitch_joint":
            dof_names.append("joint_L_ankleP")
        if name == "leg_right_hip_yaw_joint":
            dof_names.append("joint_R_hipY")
        if name == "leg_right_hip_roll_joint":
            dof_names.append("joint_R_hipR")
        if name == "leg_right_hip_pitch_joint":
            dof_names.append("joint_R_hipP")
        if name == "leg_right_knee_joint":
            dof_names.append("joint_R_knee")
        if name == "leg_right_ankle_roll_joint":
            dof_names.append("joint_R_ankleR")
        if name == "leg_right_ankle_pitch_joint":
            dof_names.append("joint_R_ankleP")
        if name == "torso_yaw_joint":
            dof_names.append("joint_torsoy1")
        if name == "torso_roll_joint":
            dof_names.append("joint_torsor")
        if name == "torso_pitch_joint":
            dof_names.append("joint_torsop")
        if name == "arm_left_shoulder_pitch_joint":
            dof_names.append("joint_arm_left_humeralp")
        if name == "arm_left_shoulder_roll_joint":
            dof_names.append("joint_arm_left_humeralr")
        if name == "arm_left_shoulder_yaw_joint":
            dof_names.append("joint_arm_left_humeraly")
        if name == "arm_left_elbow_joint":
            dof_names.append("joint_arm_left_elbow")
        if name == "arm_left_wrist_roll_joint":
            dof_names.append("joint_arm_left_wristr")
        if name == "arm_left_wrist_pitch_joint":
            dof_names.append("joint_arm_left_wristp")
        if name == "arm_left_wrist_yaw_joint":
            dof_names.append("joint_arm_left_wristy")
        if name == "arm_right_shoulder_pitch_joint":
            dof_names.append("joint_arm_right_humeralp")
        if name == "arm_right_shoulder_roll_joint":
            dof_names.append("joint_arm_right_humeralr")
        if name == "arm_right_shoulder_yaw_joint":
            dof_names.append("joint_arm_right_humeraly")
        if name == "arm_right_elbow_joint":
            dof_names.append("joint_arm_right_elbow")
        if name == "arm_right_wrist_roll_joint":
            dof_names.append("joint_arm_right_wristr")
        if name == "arm_right_wrist_pitch_joint":
            dof_names.append("joint_arm_right_wristp")
        if name == "arm_right_wrist_yaw_joint":
            dof_names.append("joint_arm_right_wristy")
       
    exp_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    policy_param = {
        "rl_version": 2.0,
        "default_joint_pos": default_joint_pos,
        "exp_name": env.cfg.experiment_name,
        "exp_time": exp_time,
        "ref_cycle_type": 0,
        "dof_names": dof_names,
        "num_actions": num_actions,
        "frame_stack": env.cfg.robot.actor_obs_history_length,
        "num_single_obs": env.num_single_obs,
        "dt": env.step_dt,
        "cycle_time": 0.8,
        "clip_observations": env.cfg.normalization.clip_observations,
        "num_observations": env.num_single_obs * env.cfg.robot.actor_obs_history_length,
        "clip_actions": env.cfg.normalization.clip_actions,
        "action_scale": env.cfg.robot.action_scale,
        "num_forcesensor": 2,

        "obs_scales_dof_pos": env.cfg.normalization.obs_scales.joint_pos,
        "obs_scales_dof_vel": env.cfg.normalization.obs_scales.joint_vel,
        "obs_scales_lin_vel": env.cfg.normalization.obs_scales.lin_vel,
        "obs_scales_ang_vel": env.cfg.normalization.obs_scales.ang_vel,
        "obs_scales_imu_waist_angvel": env.cfg.normalization.obs_scales.ang_vel,
        "obs_scales_imu_waist_orientation": env.cfg.normalization.obs_scales.projected_gravity,
        "obs_scales_torque": 0,
        "obs_scales_force": 0,
        "obs_scales_imu_torso_angvel": 0,
        "obs_scales_imu_torso_orientation": 0,

        "obs_index_commands": obs_index_commands,
        "obs_index_mode_onehot": obs_index_mode_onehot,
        "obs_index_q": obs_index_q,
        "obs_index_qd": obs_index_qd,
        "obs_index_action": obs_index_action,
        "obs_index_imu_waist_angvel": obs_index_imu_waist_angvel,
        "obs_index_imu_waist_orientation": obs_index_imu_waist_orientation,
        "obs_index_torque": obs_index_torque,
        "obs_index_forcesensor": obs_index_forcesensor,
        "obs_index_imu_waist_linearacc": obs_index_imu_waist_linearacc,
        "obs_index_imu_torso_angvel": obs_index_imu_torso_angvel,
        "obs_index_imu_torso_orientation": obs_index_imu_torso_orientation,
        "obs_index_imu_torso_linearacc": obs_index_imu_torso_linearacc,


        "has_obs_torque": (env.num_single_obs > obs_index_torque and obs_index_torque != -1),
        "has_obs_imu_waist_linearacc":
            (env.num_single_obs > obs_index_imu_waist_linearacc and obs_index_imu_waist_linearacc != -1),
        "has_obs_forcesensor":
            (env.num_single_obs > obs_index_forcesensor and obs_index_forcesensor != -1),
        "has_obs_imu_torso":
            (env.num_single_obs > obs_index_imu_torso_orientation and obs_index_imu_torso_orientation != -1),
        "has_obs_imu_torso_linearacc":
            (env.num_single_obs > obs_index_imu_torso_linearacc and obs_index_imu_torso_linearacc != -1)
    }
    return policy_param


def export_policy_param_file(env, filepath):
    params_dict = {
        "rl": policy_exported_param(env)
    }
    param_yaml = yaml.dump(params_dict)
    yaml_path = os.path.join(filepath, 'params.yaml')
    with open(yaml_path, 'w') as f:
        f.write(param_yaml)
    return yaml_path


def play():
    runner: OnPolicyRunner | AmpOnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405
    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.ranges.lin_vel_x = (0.5, 2.5)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.heading = (0.0, 0.0)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"

    #non=plane terrains for testing
    # if env_cfg.scene.terrain_generator is not None:
    #     env_cfg.scene.terrain_generator.num_rows = 5
    #     env_cfg.scene.terrain_generator.num_cols = 5
    #     env_cfg.scene.terrain_generator.curriculum = False
    #     env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    
    runner_class: OnPolicyRunner | AmpOnPolicyRunner | MapAmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    print("Runner class:", runner.__class__.__name__)

    policy = runner.get_inference_policy(device=env.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )
    param_file = export_policy_param_file(env, export_model_dir)
    print('Exported policy param to: ', param_file)
    print(f"✅ Successfully exported policy files to: {export_model_dir}")

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    obs, _ = env.get_observations()

    while simulation_app.is_running():

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)


if __name__ == "__main__":
    play()
    simulation_app.close()
