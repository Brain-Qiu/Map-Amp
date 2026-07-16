
import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
from legged_lab.envs.unitree.unitree_config_fsm import UnitreeMapAMPEnvCfg
import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sensors.camera import TiledCamera
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from legged_lab.rsl_rl.rsl_rl.env import VecEnv
from legged_lab.rsl_rl.rsl_rl.utils import AMPLoaderDisplay

from isaaclab.utils.math import quat_apply, quat_conjugate, quat_apply
from scipy.spatial.transform import Rotation

from legged_lab.envs.base.base_env_config import BaseEnvCfg
from legged_lab.utils.env_utils.scene import SceneCfg

@torch.jit.script
def copysign(a, b):
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)

@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 1, 2, 3, 0
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    half_pi = torch.tensor(np.pi / 2.0, device=q.device, dtype=q.dtype)
    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        half_pi, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)
def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz
class UnitreeEnv(VecEnv):
    def __init__(self, cfg: BaseEnvCfg, headless):
        self.cfg: BaseEnvCfg

        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed if hasattr(cfg.scene, "seed") else 42)

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]
        # Instantiate LiDAR and Depth Camera Sensors if enabled
        if self.cfg.scene.lidar.enable_lidar:
            self.lidar: RayCaster = self.scene.sensors["lidar"]
        if self.cfg.scene.depth_camera.enable_depth_camera:
            self.depth_camera: TiledCamera = self.scene.sensors["depth_camera"]

        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)

        motion_files = getattr(self.cfg, "amp_motion_files_display", [])
        if len(motion_files) > 0:
            self.amp_loader_display = AMPLoaderDisplay(
                motion_files=motion_files, device=self.device, time_between_frames=self.physics_dt
            )
            self.motion_len = self.amp_loader_display.trajectory_num_frames[0]
        else:
            self.amp_loader_display = None
            self.motion_len = 0
        # self.amp_loader_display = AMPLoaderDisplay(
        #     motion_files=self.cfg.amp_motion_files_display, device=self.device, time_between_frames=self.physics_dt
        # )
        # self.motion_len = self.amp_loader_display.trajectory_num_frames[0]

    def init_buffers(self):
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations
        print(self.robot.data.joint_names)
        self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        self.dof_pos = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.dof_names = [
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
        ]
        self.dof_ids, _ = self.robot.find_joints(
            name_keys=self.dof_names,
            preserve_order=True,
        )
        print(self.dof_ids)
        self.default_joint_pos = self.robot.data.default_joint_pos[:, self.dof_ids]
        # print(self.default_joint_pos)
        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "left_hip_pitch_joint",          
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",               
                "left_ankle_pitch_joint",       
                "left_ankle_roll_joint",
            ],
            preserve_order=True,
        )
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "right_hip_pitch_joint",          
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",               
                "right_ankle_pitch_joint",       
                "right_ankle_roll_joint",
            ],
            preserve_order=True,
        )
        self.left_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ],
            preserve_order=True,
        )
        self.right_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ],
            preserve_order=True,
        )
        self.torso_ids, _ = self.robot.find_joints(
            name_keys=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            preserve_order=True,
        )
        self.feet_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )
        self.elbow_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_elbow_link", "right_elbow_link"], preserve_order=True
        )
        self.wrist_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_wrist_yaw_link", "right_wrist_yaw_link"], preserve_order=True
        )
        self.ankle_joint_ids, _ = self.robot.find_joints(
            name_keys=["left_ankle_roll_joint", "right_ankle_roll_joint", 
                       "left_ankle_pitch_joint", "right_ankle_pitch_joint"],
            preserve_order=True,
        )
        
        # self.left_arm_local_vec = torch.tensor([0.0, 0.246, 0], device=self.device).repeat((self.num_envs, 1))
        # self.right_arm_local_vec = torch.tensor([0.0, -0.246, 0], device=self.device).repeat((self.num_envs, 1))

        self.default_dof_pos = self.robot.data.default_joint_pos[:, self.dof_ids]
        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self._run_curriculum_idx = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.action = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        # Mode state constants
        self.MODE_STAND = 0
        self.MODE_WALK = 1
        self.MODE_RUN = 2
        self.MODE_TRANSITION = 3
        
        # Common Mode Flags (for both FSM and Baseline)
        self.is_stand = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.is_move = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.is_walk = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.is_run = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.is_transition = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.current_mode = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Baseline (Ablation) specific variables
        self.last_cmd_speed = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.transition_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        # FSM Mode Manager variables
        if self.cfg.mode_manager.enable_mode_manager:
            # Intent (high-level goal): [is_stand, is_walk, is_run]
            self.intent = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
            
            # Transition target (only used when current_mode == TRANSITION)
            self.transition_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            
            # Pre-sampled target velocity for the current resampling period
            self.target_velocity = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
            
            # Time tracking
            self.mode_start_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
            self.current_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
            
            # Speed tracking
            self.actual_speed = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
            self.last_actual_speed = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
            
            # Calculate resampling steps
            resample_time = (self.cfg.commands.resampling_time_range[0] + 
                           self.cfg.commands.resampling_time_range[1]) / 2.0
            self.resample_steps = int(resample_time / self.step_dt)
            
        # print(self.robot.data.body_names)
        self.init_obs_buffer()

    def visualize_motion(self, time):
        """
        This function sets the joint positions and velocities, root position and orientation,
        and linear/angular velocities according to the AMP motion frame at the specified time,
        then steps the simulation and updates the scene.

        Args:
            time (float): The time (in seconds) at which to fetch the AMP motion frame.

        Returns:
            torch.Tensor: AMP observation tensor (70-dim)
        """
        visual_motion_frame = self.amp_loader_display.get_full_frame_at_time(0, time)
        device = self.device

        dof_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=device)
        dof_vel = torch.zeros((self.num_envs, self.robot.num_joints), device=device)
        
        dof_pos[:, self.left_leg_ids] = visual_motion_frame[6:12]
        dof_pos[:, self.right_leg_ids] = visual_motion_frame[12:18]
        dof_pos[:, self.torso_ids] = visual_motion_frame[18:21]
        dof_pos[:, self.left_arm_ids] = visual_motion_frame[21:28]
        dof_pos[:, self.right_arm_ids] = visual_motion_frame[28:35]

        dof_vel[:, self.left_leg_ids] = visual_motion_frame[41:47]
        dof_vel[:, self.right_leg_ids] = visual_motion_frame[47:53]
    
        dof_vel[:, self.torso_ids] = visual_motion_frame[53:56]
        dof_vel[:, self.left_arm_ids] = visual_motion_frame[56:63]
        dof_vel[:, self.right_arm_ids] = visual_motion_frame[63:70]

        self.robot.write_joint_position_to_sim(dof_pos)
        self.robot.write_joint_velocity_to_sim(dof_vel)

        env_ids = torch.arange(self.num_envs, device=device)

        root_pos = visual_motion_frame[:3].clone()
        euler = visual_motion_frame[3:6].cpu().numpy()
        quat_xyzw = Rotation.from_euler("XYZ", euler, degrees=False).as_quat()  # [x, y, z, w]
        quat_wxyz = torch.tensor(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=torch.float32, device=device
        )

        lin_vel = visual_motion_frame[35:38].clone()
        ang_vel = visual_motion_frame[38:41].clone() #torch.zeros_like(lin_vel)

        # root state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        root_state = torch.zeros((self.num_envs, 13), device=device)
        root_state[:, 0:3] = torch.tile(root_pos.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 3:7] = torch.tile(quat_wxyz.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 7:10] = torch.tile(lin_vel.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 10:13] = torch.tile(ang_vel.unsqueeze(0), (self.num_envs, 1))

        self.robot.write_root_state_to_sim(root_state, env_ids)
        self.sim.render()
        self.sim.step()
        self.scene.update(dt=self.step_dt)
        
        left_hand_pos = (
            self.robot.data.body_state_w[:, self.wrist_body_ids[0], :3]
            - self.robot.data.root_state_w[:, 0:3]
        )
        right_hand_pos = (
            self.robot.data.body_state_w[:, self.wrist_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
        )
        left_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos)
        right_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos)
        left_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        right_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        left_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)

        self.left_leg_dof_pos =  dof_pos[:, self.left_leg_ids] 
        self.right_leg_dof_pos = dof_pos[:, self.right_leg_ids]
        self.torso_dof_pos =     dof_pos[:, self.torso_ids]
        self.left_arm_dof_pos =  dof_pos[:, self.left_arm_ids] 
        self.right_arm_dof_pos = dof_pos[:, self.right_arm_ids]
        
        self.left_leg_dof_vel =  dof_vel[:, self.left_leg_ids] 
        self.right_leg_dof_vel = dof_vel[:, self.right_leg_ids]
        self.torso_dof_vel =     dof_vel[:, self.torso_ids]
        self.left_arm_dof_vel =  dof_vel[:, self.left_arm_ids] 
        self.right_arm_dof_vel = dof_vel[:, self.right_arm_ids]
       
        return torch.cat(
            (
                self.left_leg_dof_pos,
                self.right_leg_dof_pos,
                self.torso_dof_pos,
                self.left_arm_dof_pos,
                self.right_arm_dof_pos,
                
                self.left_leg_dof_vel,
                self.right_leg_dof_vel,
                self.torso_dof_vel,
                self.left_arm_dof_vel,
                self.right_arm_dof_vel,
    
                left_foot_pos,
                right_foot_pos,
                left_hand_pos,
                right_hand_pos,
            ),
            dim=-1,
        )

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        self.dof_vel = joint_vel[:, self.dof_ids]
        self.dof_pos = joint_pos[:, self.dof_ids]
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        base_quat = self.robot.data.root_quat_w[:, :4]
        self.base_euler_xyz = get_euler_xyz_tensor(base_quat)
        
        # Add mode one-hot if FSM is enabled
        if self.cfg.mode_manager.enable_mode_manager:
            mode_onehot = torch.zeros(self.num_envs, 4, device=self.device)
            mode_onehot[torch.arange(self.num_envs, device=self.device), self.current_mode] = 1.0
            
            current_actor_obs = torch.cat(
                [
                    command * self.obs_scales.commands,          # 3
                    mode_onehot,                                  # 4 (new)
                    self.dof_pos * self.obs_scales.joint_pos,    # 29
                    self.dof_vel * self.obs_scales.joint_vel,    # 29
                    action * self.obs_scales.actions,            # 29
                    ang_vel * self.obs_scales.ang_vel,           # 3
                    projected_gravity * self.obs_scales.projected_gravity,  # 3
                ],
                dim=-1,
            )  # Total: 100 dims
        else:
            current_actor_obs = torch.cat(
                [
                    command * self.obs_scales.commands,          # 3
                    self.dof_pos * self.obs_scales.joint_pos,    # 29
                    self.dof_vel * self.obs_scales.joint_vel,    # 29
                    action * self.obs_scales.actions,            # 29
                    ang_vel * self.obs_scales.ang_vel,           # 3
                    projected_gravity * self.obs_scales.projected_gravity,  # 3
                ],
                dim=-1,
            )  # Total: 96 dims
        self.num_single_obs = current_actor_obs.shape[-1]

        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        current_critic_obs = torch.cat(
            [current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact], dim=-1
        )

        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec
        
        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)

        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            actor_obs = torch.cat([actor_obs, height_scan], dim=-1)

        if self.cfg.scene.depth_camera.enable_depth_camera:
            depth_image = self.depth_camera.data.output["distance_to_image_plane"]
            flattened_depth = depth_image.view(self.num_envs, -1)
            actor_obs = torch.cat([actor_obs, flattened_depth], dim=-1)
            critic_obs = torch.cat([critic_obs, flattened_depth], dim=-1)
        
        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        # Reset buffer
        self.avg_feet_force_per_step[env_ids] = 0.0
        self.avg_feet_speed_per_step[env_ids] = 0.0

        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        # FSM Mode Manager: Initialize mode and intent
        if self.cfg.mode_manager.enable_mode_manager:
            self._initialize_mode_and_intent(env_ids)
            
            # FSM Mode Distribution Statistics
            total_envs = float(self.num_envs)
            mode_stats = {
                "Mode_Distribution/stand": (self.current_mode == self.MODE_STAND).sum().item() / total_envs,
                "Mode_Distribution/walk": (self.current_mode == self.MODE_WALK).sum().item() / total_envs,
                "Mode_Distribution/run": (self.current_mode == self.MODE_RUN).sum().item() / total_envs,
                "Mode_Distribution/transition": (self.current_mode == self.MODE_TRANSITION).sum().item() / total_envs,
            }
            self.extras["log"].update(mode_stats)
        
        self.command_generator.reset(env_ids)
        if not self.cfg.mode_manager.enable_mode_manager:
            # Baseline (Ablation) reset logic
            self.transition_timer[env_ids] = 0.0
            self.is_transition[env_ids] = False
            self.last_cmd_speed[env_ids] = torch.norm(self.command_generator.command[env_ids, :2], dim=-1)

        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):

        delayed_actions = self.action_buffer.compute(actions)
        self.action = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        cliped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        processed_actions = cliped_actions * self.action_scale + self.default_joint_pos
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        lab_actions = torch.zeros_like(actions)
        lab_actions[:, self.dof_ids] = processed_actions.clone()
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(lab_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            self.avg_feet_force_per_step += torch.norm(
                self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3], dim=-1
            )
            self.avg_feet_speed_per_step += torch.norm(self.robot.data.body_lin_vel_w[:, self.feet_body_ids, :], dim=-1)

        self.avg_feet_force_per_step /= self.cfg.sim.decimation
        self.avg_feet_speed_per_step /= self.cfg.sim.decimation

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        
        # FSM Mode Manager Logic
        if self.cfg.mode_manager.enable_mode_manager:
            # Update time and speed
            self.current_time += self.step_dt
            self.last_actual_speed = self.actual_speed.clone()
            self._update_actual_speed()
            
            # Check TRANSITION auto-switch (every step)
            self._check_transition_auto_switch()
            
            # Resampling: update intent and mode
            if (self.episode_length_buf % self.resample_steps == 0).any():
                resample_envs = (self.episode_length_buf % self.resample_steps == 0).nonzero(as_tuple=False).flatten()
                self._resample_intent_and_mode(resample_envs)
        
        # Update mode flags (works for both FSM and Baseline)
        self._update_mode_flags()
        
        # Log mode distribution periodically (every 1000 steps to reduce overhead)
        if self.cfg.mode_manager.enable_mode_manager:
            if self.sim_step_counter % (1000 * self.cfg.sim.decimation) == 0:
                total_envs = float(self.num_envs)
                mode_stats = {
                    "Mode_Distribution/stand": (self.current_mode == self.MODE_STAND).sum().item() / total_envs,
                    "Mode_Distribution/walk": (self.current_mode == self.MODE_WALK).sum().item() / total_envs,
                    "Mode_Distribution/run": (self.current_mode == self.MODE_RUN).sum().item() / total_envs,
                    "Mode_Distribution/transition": (self.current_mode == self.MODE_TRANSITION).sum().item() / total_envs,
                }
                if "log" not in self.extras:
                    self.extras["log"] = {}
                self.extras["log"].update(mode_stats)
                self._update_run_vx_curriculum()

        self.reset_buf, self.time_out_buf = self.check_reset()
        
        reward_buf = self.reward_manager.compute(self.step_dt)
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(self.reset_env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        reset_buf = torch.any(
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
                    dim=-1,
                ),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf
    # def check_reset(self):
    #     net_contact_forces = self.contact_sensor.data.net_forces_w_history

    #     contact_reset = torch.any(
    #     torch.max(
    #         torch.norm(
    #             net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
    #             dim=-1,
    #         ),
    #         dim=1,
    #     )[0]
    #     > 1.0,
    #     dim=1,
    #     )
    #     # base/root height termination
    #     base_height = self.robot.data.root_pos_w[:, 2]
    #     height_reset = base_height < 0.5
    #     time_out_buf = self.episode_length_buf >= self.max_episode_length
    #     reset_buf = contact_reset | height_reset | time_out_buf
    #     return reset_buf, time_out_buf

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            
            if self.cfg.mode_manager.enable_mode_manager:
                # Observation layout: [command(3), mode_onehot(4), dof_pos(29), dof_vel(29), action(29), ang_vel(3), euler(3)]
                noise_vec[:3] = 0.0  # command: no noise
                noise_vec[3:7] = 0.0  # mode_onehot: no noise
                noise_vec[7:7+self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
                noise_vec[7+self.num_actions:7+2*self.num_actions] = noise_scales.joint_vel * self.obs_scales.joint_vel
                noise_vec[7+2*self.num_actions:7+3*self.num_actions] = 0.0  # action: no noise
                noise_vec[7+3*self.num_actions:7+3*self.num_actions+3] = noise_scales.ang_vel * self.obs_scales.ang_vel
                noise_vec[7+3*self.num_actions+3:7+3*self.num_actions+6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            else:
                # Observation layout: [command(3), dof_pos(29), dof_vel(29), action(29), ang_vel(3), euler(3)]
                noise_vec[:3] = 0.0  # command: no noise
                noise_vec[3:3+self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
                noise_vec[3+self.num_actions:3+2*self.num_actions] = noise_scales.joint_vel * self.obs_scales.joint_vel
                noise_vec[3+2*self.num_actions:3+3*self.num_actions] = 0.0  # action: no noise
                noise_vec[3+3*self.num_actions:3+3*self.num_actions+3] = noise_scales.ang_vel * self.obs_scales.ang_vel
                noise_vec[3+3*self.num_actions+3:3+3*self.num_actions+6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            
            self.noise_scale_vec = noise_vec

            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )

    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {"Curriculum/terrain_levels": torch.mean(self.scene.terrain.terrain_levels.float())}
        return extras

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, self.extras

    def get_amp_obs_for_expert_trans(self):
        """Gets amp obs from policy )"""
        left_hand_pos = (
            self.robot.data.body_state_w[:, self.wrist_body_ids[0], :3]
            - self.robot.data.root_state_w[:, 0:3]
        )
        right_hand_pos = (
            self.robot.data.body_state_w[:, self.wrist_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
        )
        left_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos)
        right_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos)
        left_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        right_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        left_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)
        
        self.left_leg_dof_pos = self.robot.data.joint_pos[:, self.left_leg_ids]    # 6维
        self.right_leg_dof_pos = self.robot.data.joint_pos[:, self.right_leg_ids]  # 6维
        self.torso_dof_pos = self.robot.data.joint_pos[:, self.torso_ids]          # 3维
        self.left_arm_dof_pos = self.robot.data.joint_pos[:, self.left_arm_ids]    # 7维
        self.right_arm_dof_pos = self.robot.data.joint_pos[:, self.right_arm_ids]  # 7维
        
        self.left_leg_dof_vel = self.robot.data.joint_vel[:, self.left_leg_ids]    # 6维
        self.right_leg_dof_vel = self.robot.data.joint_vel[:, self.right_leg_ids]  # 6维
        self.torso_dof_vel = self.robot.data.joint_vel[:, self.torso_ids]          # 3维
        self.left_arm_dof_vel = self.robot.data.joint_vel[:, self.left_arm_ids]    # 7维
        self.right_arm_dof_vel = self.robot.data.joint_vel[:, self.right_arm_ids]  # 7维
        
        return torch.cat(
            (   
                self.left_leg_dof_pos,
                self.right_leg_dof_pos,
                self.torso_dof_pos,
                self.left_arm_dof_pos,
                self.right_arm_dof_pos,
                
                self.left_leg_dof_vel,
                self.right_leg_dof_vel,
                self.torso_dof_vel,
                self.left_arm_dof_vel,
                self.right_arm_dof_vel,
    
                left_foot_pos,
                right_foot_pos,
                left_hand_pos,
                right_hand_pos,
            ),
            dim=-1,
        )  # 29+29+12 = 70维
    
    # ========== FSM Mode Manager Methods ==========
    
    def _update_actual_speed(self):
        """Update current speed (xy plane norm)"""
        if not self.cfg.mode_manager.enable_mode_manager:
            return
        vel_body = self.robot.data.root_lin_vel_b
        self.actual_speed = torch.norm(vel_body[:, :2], dim=1)
    
    def _sample_intent(self, env_ids):
        """Sample intent for specified environments"""
        if not self.cfg.mode_manager.enable_mode_manager:
            return
            
        n = len(env_ids)
        rand = torch.rand(n, device=self.device)
        
        self.intent[env_ids] = 0.0  # Reset
        
        # Distribute intents based on configuration
        stand_mask = rand < self.cfg.mode_manager.rel_stand_envs
        walk_mask = (rand >= self.cfg.mode_manager.rel_stand_envs) & \
                    (rand < self.cfg.mode_manager.rel_stand_envs + self.cfg.mode_manager.rel_walk_envs)
        run_mask = rand >= (self.cfg.mode_manager.rel_stand_envs + self.cfg.mode_manager.rel_walk_envs)
        
        self.intent[env_ids[stand_mask], 0] = 1.0  # Stand
        self.intent[env_ids[walk_mask], 1] = 1.0   # Walk
        self.intent[env_ids[run_mask], 2] = 1.0    # Run
        
        # Sample target velocities for each intent
        # Stand
        self.target_velocity[env_ids[stand_mask]] = 0.0
        
        # Walk
        n_walk = walk_mask.sum().item()
        if n_walk > 0:
            walk_ids = env_ids[walk_mask]
            vx = torch.empty(n_walk, device=self.device).uniform_(*self.cfg.mode_manager.walk_vel_x_range)
            vy = torch.empty(n_walk, device=self.device).uniform_(*self.cfg.mode_manager.walk_vel_y_range)
            vyaw = torch.empty(n_walk, device=self.device).uniform_(*self.cfg.mode_manager.walk_vel_yaw_range)
            self.target_velocity[walk_ids] = torch.stack([vx, vy, vyaw], dim=1)
        
        # Run
        n_run = run_mask.sum().item()
        if n_run > 0:
            run_ids = env_ids[run_mask]
            vx = torch.empty(n_run, device=self.device).uniform_(*self.cfg.mode_manager.run_vel_x_range)
            vy = torch.empty(n_run, device=self.device).uniform_(*self.cfg.mode_manager.run_vel_y_range)
            vyaw = torch.empty(n_run, device=self.device).uniform_(*self.cfg.mode_manager.run_vel_yaw_range)
            self.target_velocity[run_ids] = torch.stack([vx, vy, vyaw], dim=1)
    
    def _initialize_mode_and_intent(self, env_ids):
        """Smart initialization of mode and intent at episode start"""
        if not self.cfg.mode_manager.enable_mode_manager:
            return
        # 1. Sample intents
        self._sample_intent(env_ids) 
        # 2. Smart initialization of current mode
        if self.cfg.mode_manager.smart_initialization:
            for i in env_ids:
                intent_id = self.intent[i].argmax().item()
                rand = torch.rand(1).item()
                
                if intent_id == 0:  # STAND intent
                    # Balanced initialization to train all deceleration paths
                    # 34% RUN→STAND, 33% WALK→STAND, 33% STAND
                    if rand < 0.20:
                        self.current_mode[i] = self.MODE_RUN    # Critical: trains RUN→STAND transition
                    elif rand < 0.60:
                        self.current_mode[i] = self.MODE_WALK   # Trains WALK→STAND
                    else:
                        self.current_mode[i] = self.MODE_STAND  # Already at target
                
                elif intent_id == 1:  # WALK intent
                    # Balanced initialization: 34% STAND, 33% WALK, 33% RUN
                    if rand < 0.20:
                        self.current_mode[i] = self.MODE_STAND  # STAND→WALK acceleration
                    elif rand < 0.60:
                        self.current_mode[i] = self.MODE_WALK   # Stay in WALK
                    else:
                        self.current_mode[i] = self.MODE_RUN    # RUN→WALK deceleration
                
                elif intent_id == 2:  # RUN intent
                    # Balanced initialization: 34% STAND, 33% WALK, 33% RUN
                    if rand < 0.20:
                        self.current_mode[i] = self.MODE_STAND  # STAND→TRANSITION→RUN
                    elif rand < 0.60:
                        self.current_mode[i] = self.MODE_WALK   # WALK→RUN direct
                    else:
                        self.current_mode[i] = self.MODE_RUN    # Start from RUN
        else:
            # Random initialization (only STAND or WALK, no RUN)
            rand = torch.rand(len(env_ids), device=self.device)
            self.current_mode[env_ids] = torch.where(
                rand < 0.5,
                torch.tensor(self.MODE_STAND, device=self.device),
                torch.tensor(self.MODE_WALK, device=self.device)
            )
        
        # 3. Reset mode start time
        self.mode_start_time[env_ids] = self.current_time[env_ids]
        
        # 4. Update mode based on intent
        self._update_mode_based_on_intent(env_ids)
    
    def _check_transition_auto_switch(self):
        """Check and perform automatic mode switching for TRANSITION state"""
        if not self.cfg.mode_manager.enable_mode_manager:
            return
        
        trans_mask = (self.current_mode == self.MODE_TRANSITION)
        if trans_mask.sum() == 0:
            return
        
        threshold = self.cfg.mode_manager.transition_speed_threshold
        tolerance = self.cfg.mode_manager.speed_tolerance
        
        # Check TRANSITION → RUN (acceleration complete)
        to_run_mask = trans_mask & (self.transition_target == self.MODE_RUN)
        speed_reached = self.actual_speed >= (threshold - tolerance)
        switch_to_run = to_run_mask & speed_reached
        
        if switch_to_run.sum() > 0:
            env_ids = switch_to_run.nonzero(as_tuple=False).flatten()
            self.current_mode[env_ids] = self.MODE_RUN
            self.mode_start_time[env_ids] = self.current_time[env_ids]
            # Update cmd_vel to target velocity
            self.command_generator.command[env_ids] = self.target_velocity[env_ids]
        
        # Check TRANSITION → STAND (deceleration complete)
        to_stand_mask = trans_mask & (self.transition_target == self.MODE_STAND)
        # Speed must be descending and below threshold
        speed_decreasing = self.actual_speed < self.last_actual_speed
        speed_reached = self.actual_speed <= (threshold + tolerance)
        switch_to_stand = to_stand_mask & speed_reached & speed_decreasing
        
        if switch_to_stand.sum() > 0:
            env_ids = switch_to_stand.nonzero(as_tuple=False).flatten()
            self.current_mode[env_ids] = self.MODE_STAND
            self.mode_start_time[env_ids] = self.current_time[env_ids]
            # Update cmd_vel to zero
            self.command_generator.command[env_ids] = 0.0
    
    def _can_switch_mode(self, env_ids):
        """Check if environments can switch mode (min duration check)"""
        if not self.cfg.mode_manager.enable_mode_manager:
            return torch.ones(len(env_ids), device=self.device, dtype=torch.bool)
        
        # TRANSITION mode can always switch (no min duration)
        is_transition = self.current_mode[env_ids] == self.MODE_TRANSITION
        
        # For other modes, check min duration
        duration = self.current_time[env_ids] - self.mode_start_time[env_ids]
        duration_ok = duration >= self.cfg.mode_manager.min_mode_duration
        
        return is_transition | duration_ok
    
    def _resample_intent_and_mode(self, env_ids):
        """Resample intent and update mode (called every resampling period)"""
        if not self.cfg.mode_manager.enable_mode_manager:
            return
        
        # Re-sample intent for all environments
        self._sample_intent(env_ids)
        
        # Update mode based on new intent
        self._update_mode_based_on_intent(env_ids)
    
    def _update_mode_based_on_intent(self, env_ids):
        """FSM core logic: update mode and cmd_vel based on intent"""
        if not self.cfg.mode_manager.enable_mode_manager:
            return
        
        # Check which environments can switch mode
        can_switch = self._can_switch_mode(env_ids)
        
        for idx, i in enumerate(env_ids):
            if not can_switch[idx]:
                continue  # Skip if min duration not met
            
            intent_id = self.intent[i].argmax().item()
            current = self.current_mode[i].item()
            
            # Read pre-sampled target velocity for the intent
            target_vel = self.target_velocity[i].clone()
            
            # FSM transition logic
            if current == self.MODE_STAND:
                if intent_id == 0:  # STAND → STAND
                    # Keep
                    pass
                elif intent_id == 1:  # STAND → WALK
                    self.current_mode[i] = self.MODE_WALK
                    self.mode_start_time[i] = self.current_time[i]
                    self.command_generator.command[i] = target_vel
                elif intent_id == 2:  # STAND → RUN (via TRANSITION)
                    self.current_mode[i] = self.MODE_TRANSITION
                    self.transition_target[i] = self.MODE_RUN
                    self.mode_start_time[i] = self.current_time[i]
                    # Set transition speed (1.0 m/s)
                    self.command_generator.command[i] = torch.tensor(
                        [self.cfg.mode_manager.transition_speed_threshold, 0.0, 0.0],
                        device=self.device
                    )
            
            elif current == self.MODE_WALK:
                if intent_id == 0:  # WALK → STAND
                    self.current_mode[i] = self.MODE_STAND
                    self.mode_start_time[i] = self.current_time[i]
                    self.command_generator.command[i] = 0.0
                elif intent_id == 1:  # WALK → WALK
                    # Re-sample velocity
                    self.command_generator.command[i] = target_vel
                elif intent_id == 2:  # WALK → RUN
                    self.current_mode[i] = self.MODE_RUN
                    self.mode_start_time[i] = self.current_time[i]
                    self.command_generator.command[i] = target_vel
            
            elif current == self.MODE_RUN:
                if intent_id == 0:  # RUN → STAND (via TRANSITION)
                    self.current_mode[i] = self.MODE_TRANSITION
                    self.transition_target[i] = self.MODE_STAND
                    self.mode_start_time[i] = self.current_time[i]
                    # Set transition speed (1.0 m/s)
                    self.command_generator.command[i] = torch.tensor(
                        [self.cfg.mode_manager.transition_speed_threshold, 0.0, 0.0],
                        device=self.device
                    )
                elif intent_id == 1:  # RUN → WALK
                    self.current_mode[i] = self.MODE_WALK
                    self.mode_start_time[i] = self.current_time[i]
                    self.command_generator.command[i] = target_vel
                elif intent_id == 2:  # RUN → RUN
                    # Re-sample velocity
                    self.command_generator.command[i] = target_vel
            
            elif current == self.MODE_TRANSITION:
                # TRANSITION can immediately respond to new intent
                old_target = self.transition_target[i].item()
                
                if intent_id == old_target:
                    # Intent matches current target, continue
                    continue
                
                # Intent changed during transition
                if intent_id == 1:  # Any → WALK
                    # Can directly switch to WALK
                    self.current_mode[i] = self.MODE_WALK
                    self.mode_start_time[i] = self.current_time[i]
                    self.command_generator.command[i] = target_vel
                
                elif intent_id == 0:  # → STAND
                    # Continue TRANSITION but change target to STAND
                    self.transition_target[i] = self.MODE_STAND
                    self.command_generator.command[i] = torch.tensor(
                        [self.cfg.mode_manager.transition_speed_threshold, 0.0, 0.0],
                        device=self.device
                    )
                
                elif intent_id == 2:  # → RUN
                    # Continue TRANSITION but change target to RUN
                    self.transition_target[i] = self.MODE_RUN
                    self.command_generator.command[i] = torch.tensor(
                        [self.cfg.mode_manager.transition_speed_threshold, 0.0, 0.0],
                        device=self.device
                    )
    
    def _update_mode_flags(self):
        """Update global mode flags for reward functions"""
        if not self.cfg.mode_manager.enable_mode_manager:
            # Baseline (Ablation) logic based on command speed with hard transition detection
            cmd_speed = torch.norm(self.command_generator.command[:, :2], dim=-1)
            
            # Detect jumps (transition trigger)
            cmd_jump = torch.abs(cmd_speed - self.last_cmd_speed) >= 1.5
            
            # Update transition state and timer
            self.is_transition = self.is_transition | cmd_jump
            self.transition_timer[cmd_jump] = 0.0
            
            # Transition timeout condition (hard switch at 2.0s)
            timeout = self.transition_timer > 2.0
            
            # End transition if timeout
            self.is_transition = self.is_transition & (~timeout)
            
            # Advance timer for items still in transition
            self.transition_timer[self.is_transition] += self.step_dt
            
            # Steady states based purely on cmd_speed
            is_cmd_stand = cmd_speed < 0.1
            is_cmd_walk = (cmd_speed >= 0.1) & (cmd_speed < 1.5)
            is_cmd_run = cmd_speed >= 1.5
            
            # Final flags (suppressed by transition)
            self.is_stand = (~self.is_transition) & is_cmd_stand
            self.is_walk = (~self.is_transition) & is_cmd_walk
            self.is_run = (~self.is_transition) & is_cmd_run
            
            self.is_move = self.is_walk | self.is_run | self.is_transition  
            
            # Sync to current_mode
            self.current_mode[self.is_stand] = self.MODE_STAND
            self.current_mode[self.is_walk] = self.MODE_WALK
            self.current_mode[self.is_run] = self.MODE_RUN
            self.current_mode[self.is_transition] = self.MODE_TRANSITION
            
            # Save for next step comparison
            self.last_cmd_speed = cmd_speed.clone()
        else:
            # Use FSM mode
            self.is_stand = (self.current_mode == self.MODE_STAND)
            self.is_walk = (self.current_mode == self.MODE_WALK)
            self.is_run = (self.current_mode == self.MODE_RUN)
            self.is_transition = (self.current_mode == self.MODE_TRANSITION)
            self.is_move = self.is_walk | self.is_run  | self.is_transition
    
    def _update_run_vx_curriculum(self):
        """Advance run_vel_x_range upper bound per the iteration-based schedule."""
        schedule = self.cfg.mode_manager.run_vx_curriculum
        if not schedule or self._run_curriculum_idx >= len(schedule):
            return
        steps_per_iter = self.cfg.mode_manager.run_vx_curriculum_steps_per_iter
        policy_iter = self.sim_step_counter // (self.cfg.sim.decimation * steps_per_iter)
        trigger_iter, new_vx_max = schedule[self._run_curriculum_idx]
        if policy_iter >= trigger_iter:
            vx_min = self.cfg.mode_manager.run_vel_x_range[0]
            self.cfg.mode_manager.run_vel_x_range = (vx_min, new_vx_max)
            self._run_curriculum_idx += 1
            if "log" not in self.extras:
                self.extras["log"] = {}
            self.extras["log"]["Curriculum/run_vx_max"] = new_vx_max
            print(f"[RunCurriculum] iter?{policy_iter}: run_vel_x_range ? ({vx_min:.1f}, {new_vx_max:.1f}) m/s")
    
    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)
