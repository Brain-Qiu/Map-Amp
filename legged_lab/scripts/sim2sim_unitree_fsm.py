"""sim2sim without ROS dependencies and config files (Unitree model compatibility)."""

import math
import os
import sys
import time
from collections import deque

workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(workspace_dir)

import mujoco
import mujoco.viewer
import numpy as np
from pynput import keyboard
from scipy.spatial.transform import Rotation as R
import torch
import yaml

G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
]
G1_KPS = [
    100, 100, 100, 150, 40, 40,  # Left Leg  (hip_p/r/y, knee, ankle_p/r)
    100, 100, 100, 150, 40, 40,  # Right Leg (hip_p/r/y, knee, ankle_p/r)
    200, 40, 40,                 # Waist     (yaw, roll, pitch)
    40, 40, 40, 40, 40, 40, 40,  # Left Arm  (shoulder_p/r/y, elbow, wrist_r/p/y)
    40, 40, 40, 40, 40, 40, 40   # Right Arm (shoulder_p/r/y, elbow, wrist_r/p/y)
]

G1_KDS = [
    2, 2, 2, 4, 2, 2,  # Left Leg
    2, 2, 2, 4, 2, 2,  # Right Leg
    5, 5, 5,           # Waist
    1, 1, 1, 1, 1, 1, 1,  # Left Arm
    1, 1, 1, 1, 1, 1, 1   # Right Arm
]

G1_TAU_LIMITS = [
    88, 88, 88, 139, 50, 50,  # Left Leg
    88, 88, 88, 139, 50, 50,  # Right Leg
    88, 25, 25,                # Waist
    25, 25, 25, 25, 25, 5, 5,  # Left Arm
    25, 25, 25, 25, 25, 5, 5   # Right Arm
]
# -------------------------------------------------------------------

class cmd:
    """Robot command."""
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

stand_flag = 1

def on_key_press(key):
    """Get key press."""
    try:
        ch = key.char.lower()
        if ch not in 'wsadkfqe2467890':
            return
        keyboard_control(ch)
    except AttributeError:
        pass

def keyboard_control(key):
    """."""
    if key in ['w', '8']:
        cmd.vx = cmd.vx + 0.2
        if cmd.vx > 3.5: cmd.vx = 3.5
    if key in ['s', '2']:
        cmd.vx = cmd.vx - 0.2
        if cmd.vx < -0.6: cmd.vx = -0.6
    if key in ['a', '6']:
        cmd.dyaw = cmd.dyaw - 0.1
        if cmd.dyaw < -1.0: cmd.dyaw = -1.0
    if key in ['d', '4']:
        cmd.dyaw = cmd.dyaw + 0.1
        if cmd.dyaw > 1.0: cmd.dyaw = 1.0
    if key in ['q', '7']:
        cmd.vy = cmd.vy + 0.2
        if cmd.vy > 0.6: cmd.vy = 0.6
    if key in ['e', '9']:
        cmd.vy = cmd.vy - 0.2
        if cmd.vy < -0.6: cmd.vy = -0.6
    if key in ['k', '0']:
        cmd.vx = 0; cmd.vy = 0; cmd.dyaw = 0
    if key == 'f':
        global stand_flag
        stand_flag *= -1
    # print(f"Commands: vx={cmd.vx:.2f}, vy={cmd.vy:.2f}, dyaw={cmd.dyaw:.2f}\r")


def quaternion_to_rotation_matrix(q):
    q /= np.linalg.norm(q)
    x, y, z, w = q
    R_mat = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
    ])
    return R_mat


class HumaRobotSim:
    def __init__(self) -> None:
        self.rl_last_time = None
        self._gait_time = 0.0
        self.action_delay = 0.0

        # FSM State
        self.current_mode = 0  # 0: STAND, 1: WALK, 2: RUN, 3: TRANSITION
        self.transition_target = -1
        self.target_velocity = np.zeros(3)
        self.last_actual_speed = 0.0

    def get_obs(self, data):
        q = data.qpos.astype(np.double)
        dq = data.qvel.astype(np.double)
        
        # Base orientation from qpos[3:7] ([w,x,y,z]) mapped to [x,y,z,w] for scipy R

        quat = q[3:7][[1, 2, 3, 0]].astype(np.double)

        # ################ debug#################
        # euler = R.from_quat(quat).as_euler("XYZ", degrees=False)
        # euler_bias = np.array([0.0, -0.06, 0.0], dtype=np.double)
        # euler = euler + euler_bias
        # quat = R.from_euler("XYZ", euler, degrees=False).as_quat().astype(np.double)
        # #####################debug##############

        r = R.from_quat(quat)
        
        # Local velocity derived from qvel 
        v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
        lin_vel = v.copy() 

        # IMU sensors based directly on the g1_29dof.xml site sensors
        omega = data.sensor('imu-pelvis-angular-velocity').data.astype(np.double)
        lin_acc = data.sensor('imu-pelvis-linear-acceleration').data.astype(np.double)
        gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
        
        # Unitree XML model in use doesn't map foot forces realistically to these variables
        torque_foot = np.zeros([4], dtype=np.double)
        force_foot  = np.zeros([2], dtype=np.double)
        src_force_foot = np.zeros([2], dtype=np.double)
        force_sensor = np.zeros([6], dtype=np.double)

        return (q, dq, quat, v, omega, gvec, lin_acc, torque_foot, lin_vel, force_foot, src_force_foot, force_sensor)

    def pd_control(self, target_q, q, kp, target_dq, dq, kd):
        return (target_q - q) * kp + (target_dq - dq) * kd

    def gen_rl_clock(self):
        cur_time = time.time()
        if self.rl_last_time is None:
            period = 0.0
        else:
            period = cur_time - self.rl_last_time
        self.rl_last_time = cur_time

        self._gait_time += period
        if self._gait_time > self.step_sycle_time:
            self._gait_time -= self.step_sycle_time
        phase = self._gait_time / self.step_sycle_time
        
        if self.param.get('ref_cycle_type', 0) == 1:
            clock_a = math.sin(2 * math.pi * phase)
            clock_b = math.sin(2 * math.pi * phase)
            if clock_a > 0: clock_a = 0
            else: clock_a = -clock_a
            if clock_a < 0.1: clock_a = 0
            if clock_b < 0: clock_b = 0
            if clock_b < 0.1: clock_b = 0
        else:
            clock_a = math.sin(2 * math.pi * phase)
            clock_b = math.cos(2 * math.pi * phase)
        return clock_a, clock_b, phase

    def rl_policy_forward(self, param, policy, q, dq, waist_orient_quat, waist_omega, torque_foot, force_foot, lin_vel, gvec):
        obs = np.zeros([1, param['num_single_obs']], dtype=np.float32)
        
        self.waist_lin_acc_from_vel = (self.old_lin_vel - lin_vel) / 0.01
        self.old_lin_vel = lin_vel.copy()
        
        self.gen_rl_clock()
        
        if 'obs_index_mode_onehot' in param:
            intent_speed = np.sqrt(cmd.vx**2 + cmd.vy**2)
            if intent_speed < 0.1:
                intent_id = 0  # STAND
            elif intent_speed < 1.5:
                intent_id = 1  # WALK
            else:
                intent_id = 2  # RUN
            
            intent_vel = np.array([cmd.vx, cmd.vy, cmd.dyaw])
            actual_speed = np.linalg.norm(lin_vel[:2])
            self.last_actual_speed = actual_speed

            if self.current_mode == 3:  # TRANSITION
                if intent_id != self.transition_target:
                    self.current_mode = intent_id
                    self.transition_target = -1
                    self.target_velocity = intent_vel
                elif self.transition_target == 2 and actual_speed >= 1.5:
                    # STAND -> RUN 
                    self.current_mode = self.transition_target
                    self.transition_target = -1
                    self.target_velocity = intent_vel
                elif self.transition_target == 0 and actual_speed <= 1.4:
                    # RUN -> STAND 
                    self.current_mode = self.transition_target
                    self.transition_target = -1
                    self.target_velocity = intent_vel
                else:
                    if self.transition_target == 2:
                        self.target_velocity = intent_vel.copy()
                        self.target_velocity[:2] = (intent_vel[:2] / (intent_speed + 1e-6)) * 1.5
                    elif self.transition_target == 0:
                        self.target_velocity = np.zeros(3)
                        cur_vel_dir = lin_vel[:2] / (actual_speed + 1e-6)
                        self.target_velocity[:2] = cur_vel_dir * 1.0
            else:
                if self.current_mode == 0 and intent_id == 2:
                    self.current_mode = 3
                    self.transition_target = 2
                    self.target_velocity = intent_vel.copy()
                    self.target_velocity[:2] = (intent_vel[:2] / (intent_speed + 1e-6)) * 1.5
                elif self.current_mode == 2 and intent_id == 0:
                    #  RUN -> STAND 
                    self.current_mode = 3
                    self.transition_target = 0
                    self.target_velocity = np.zeros(3)
                    cur_vel_dir = lin_vel[:2] / (actual_speed + 1e-6)
                    self.target_velocity[:2] = cur_vel_dir * 1.0
                else:
                    self.current_mode = intent_id
                    self.target_velocity = intent_vel
                    
            # Update commands based on FSM state
            commands = [self.target_velocity[0] * param['obs_scales_lin_vel'], self.target_velocity[1] * param['obs_scales_lin_vel'], self.target_velocity[2] * param['obs_scales_ang_vel']]
            obs[0, param['obs_index_commands'] : param['obs_index_commands'] + len(commands)] = commands
            
            # Generate one-hot encoding
            mode_onehot = np.zeros(4, dtype=np.float32)
            mode_onehot[self.current_mode] = 1.0
            
            # Fill observation
            obs[0, param['obs_index_mode_onehot']:param['obs_index_mode_onehot'] + 4] = mode_onehot
        else:
            commands = [cmd.vx * param['obs_scales_lin_vel'], cmd.vy * param['obs_scales_lin_vel'], cmd.dyaw * param['obs_scales_ang_vel']]
            obs[0, param['obs_index_commands'] : param['obs_index_commands'] + len(commands)] = commands
        
        obs[0, param['obs_index_q'] : param['obs_index_q'] + len(q)] = q * param['obs_scales_dof_pos']
        obs[0, param['obs_index_qd'] : param['obs_index_qd'] + len(dq)] = dq * param['obs_scales_dof_vel']
        obs[0, param['obs_index_action'] : param['obs_index_action'] + len(self.action)] = self.action
        
        obs[0, param['obs_index_imu_waist_angvel'] : param['obs_index_imu_waist_angvel'] + len(waist_omega)] =\
            waist_omega * param['obs_scales_imu_waist_angvel']
        obs[0, param['obs_index_imu_waist_orientation'] : param['obs_index_imu_waist_orientation'] + len(gvec)] =\
            gvec * param['obs_scales_imu_waist_orientation']

        obs = np.clip(obs, -param.get('clip_observations', 100), param.get('clip_observations', 100))
        self.hist_obs.append(obs)

        policy_input = np.zeros([1, param['num_observations']], dtype=np.float32)
        for i in range(param['frame_stack']):
            policy_input[0, i * param['num_single_obs'] : (i + 1) * param['num_single_obs']] = self.hist_obs[i][0, :]

        with torch.no_grad():
            self.action[:] = policy(torch.tensor(policy_input))[0].numpy()

        self.action = np.clip(self.action, -param.get('clip_actions', 100), param.get('clip_actions', 100))
        self.action = self.action_delay * self.last_action + (1 - self.action_delay) * self.action
        self.last_action = self.action.copy()
        
        target_q = self.action * param['action_scale']
        return target_q

    def run_mujoco(self, policy, param, mjcf_path):
        self.param = param
        self.policy = policy
        self.dof = param['num_actions']  # Should be 29 for G1
        
        model = mujoco.MjModel.from_xml_path(mjcf_path)
        sim_dt = 0.001 
        model.opt.timestep = sim_dt
        
        model.dof_armature[-self.dof:] = 0.01 #same with train_env 0.01

        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        self.default_joint_pos = np.array(param['default_joint_pos'], dtype=np.double)
        print(f"Loaded default joint pos: {self.default_joint_pos}")
        data.qpos[-self.dof:] = self.default_joint_pos
        data.qpos[2] = 0.74     #0.74
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        policy_dt = param.get('dt', 0.02)
        decimation = int(policy_dt / sim_dt)
        print(f"Loaded XML: {mjcf_path}")
        print(f"Policy dt: {policy_dt}s, Sim dt: {sim_dt}s, Sim decimation: {decimation}")
        
        target_q = np.zeros((self.dof), dtype=np.double)
        self.action = np.zeros((self.dof), dtype=np.double)
        self.last_action = np.zeros((self.dof), dtype=np.double)

        self.old_lin_vel = np.zeros([3], dtype=np.double)
        
        self.hist_obs = deque(maxlen=param['frame_stack'])
        for _ in range(param['frame_stack']):
            self.hist_obs.append(np.zeros([1, param['num_single_obs']], dtype=np.double))
            
        count_lowlevel = 0
        
        pd_kps = np.array(G1_KPS, dtype=np.double)
        pd_kds = np.array(G1_KDS, dtype=np.double)
        tau_limit = np.array(G1_TAU_LIMITS, dtype=np.double)

        ref_time = time.time()
        self.tau = np.zeros((self.dof), dtype=np.double)
        self.step_sycle_time = param.get('cycle_time', 0.8)
        
        print(f"Starting Simulation with DOF: {self.dof}...")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                q, dq, quat, v, omega, gvec, lin_acc, torque_foot, lin_vel, force_foot, src_frc_foot, force_sensor = self.get_obs(data)
                q_joints = q[-self.dof:]
                dq_joints = dq[-self.dof:]
                
                rBody = quaternion_to_rotation_matrix(quat)
                force = np.zeros([12])
                force[:6] = force_sensor

                if count_lowlevel % decimation == 0:
                    print(
                        f"[Actual]   vx={v[0]:+.3f}  vy={v[1]:+.3f}  yaw={omega[2]:+.3f}      \n"
                        f"[Cmd]      vx={cmd.vx:+.3f}  vy={cmd.vy:+.3f}  yaw={cmd.dyaw:+.3f}      \033[1A\r",
                        end="",
                    )
                    target_q = self.rl_policy_forward(param, policy, q_joints - self.default_joint_pos, dq_joints, quat, omega, torque_foot, force_foot, lin_vel, gvec)
                    target_q = target_q + self.default_joint_pos 

                target_dq = np.zeros((self.dof), dtype=np.double)
                
                self.tau = self.pd_control(target_q, q_joints, pd_kps, target_dq, dq_joints, pd_kds)
                self.tau = np.clip(self.tau, -tau_limit, tau_limit)
                data.ctrl = self.tau

                if rBody[2, 2] < 0.1:
                    mujoco.mj_resetData(model, data)
                mujoco.mj_step(model, data)
                ref_time += model.opt.timestep

                if count_lowlevel % decimation == 0:
                    viewer.sync()
                    cur_time = time.time()
                    delta_time_s = ref_time - cur_time
                    if delta_time_s > 0:
                        time.sleep(delta_time_s)

                count_lowlevel += 1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Deployment script specific for Unitree model without ROS or config files.')
    parser.add_argument('--experiment_name', type=str, default='unitree_rl', help='Experiment log folder name.')
    parser.add_argument('--load_run', type=str, default='example', help='Directory name inside experiment folder.')
    
    args = parser.parse_args()

    if args.load_run == 'example':
        # without --load_run, use the policy in the example/ folder
        run_dir = os.path.join(workspace_dir, "example")
        model_path = os.path.join(run_dir, "policy.pt")
        param_path = os.path.join(run_dir, "params.yaml")
    else:
        base_log_dir = os.path.join(workspace_dir, "logs", args.experiment_name)
        run_dir = os.path.join(base_log_dir, args.load_run)
        model_path = os.path.join(run_dir, "exported", "policy.pt")
        param_path = os.path.join(run_dir, "exported", "params.yaml")
    
    mjcf_path = os.path.join(workspace_dir, "legged_lab", "assets", "unitree_fsm", "mjcf", "g1_29dof.xml")

    print(f"XML Path  : {mjcf_path}")
    print(f"Model Path: {model_path}")
    print(f"Param Path: {param_path}")

    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()
    
    policy = torch.jit.load(model_path)
    with open(param_path, 'r') as yaml_file:
        data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    param = data["rl"]
    
    # FSM Diagnostic Information
    print("\n" + "="*60)
    if 'obs_index_mode_onehot' in param:
        print("✅ FSM Mode Detection: mode_onehot_index = {}".format(param['obs_index_mode_onehot']))
        print(" Mode Inference Logic:")
        print("   speed < 0.1 m/s      → STAND  (mode=0)")
        print("   0.1 ≤ speed < 1.5    → WALK   (mode=1)")
        print("   speed ≥ 1.5 m/s      → RUN    (mode=2)")
    else:
        print("⚠️  Non-FSM Mode: obs_index_q = {} (should be 3, no mode_onehot)".format(param['obs_index_q']))
        print("   num_single_obs = {} (standard: 96 dims or similar)".format(param['num_single_obs']))
    print("="*60 + "\n")
    
    sim = HumaRobotSim()
    sim.run_mujoco(policy, param, mjcf_path)
