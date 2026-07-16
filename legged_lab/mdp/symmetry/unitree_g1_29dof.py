"""Functions to specify the symmetry in the observation and action space for Unitree G1 (29 DOF)."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv

__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: BaseEnv,
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "policy",
):
    """Augments the given observations and actions by applying symmetry transformations.

    Args:
        env: The environment instance.
        obs: The original observation tensor. Defaults to None.
        actions: The original actions tensor. Defaults to None.
        obs_type: "policy" or "critic".

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """
    if obs is not None:
        batch_size = obs.shape[0]
        obs_aug = obs.repeat(2, 1)
        obs_aug[:batch_size] = obs[:]
        if obs_type == "policy":
            obs_aug[batch_size : 2 * batch_size] = _transform_policy_obs_left_right(env, obs)
        elif obs_type == "critic":
            obs_aug[batch_size : 2 * batch_size] = _transform_critic_obs_left_right(env, obs)
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


# ============================================================
# Observation transformations
# ============================================================

def _transform_policy_obs_left_right(env: BaseEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply left-right symmetry to the policy observation tensor.

    Policy observation layout (with FSM mode manager, 29 DOF):
        [0:3]   command        (3)
        [3:7]   mode_onehot    (4)
        [7:36]  joint_pos      (29)
        [36:65] joint_vel      (29)
        [65:94] last_actions   (29)
        [94:97] ang_vel        (3)
        [97:100] proj_gravity  (3)
        Total: 100 dims per frame
    """
    obs = obs.clone()
    device = obs.device
    obs_dim = 100
    assert obs.shape[1] % obs_dim == 0, \
        f"Obs dim {obs.shape[1]} is not divisible by {obs_dim}"
    num_frames = obs.shape[1] // obs_dim
    obs = obs.view(obs.shape[0], num_frames, obs_dim)

    # velocity command: x stays, y and yaw flip
    obs[..., 0:3] = obs[..., 0:3] * torch.tensor([1, -1, -1], device=device)
    # mode one-hot [3:7]: symmetric under left-right flip, unchanged
    # joint_pos, joint_vel, actions
    obs[..., 7:36]  = _switch_joints_left_right(obs[..., 7:36])
    obs[..., 36:65] = _switch_joints_left_right(obs[..., 36:65])
    obs[..., 65:94] = _switch_joints_left_right(obs[..., 65:94])
    # angular velocity: roll and yaw flip, pitch stays
    obs[..., 94:97] = obs[..., 94:97] * torch.tensor([-1, 1, -1], device=device)
    # projected gravity: x and z stay, y flips
    obs[..., 97:100] = obs[..., 97:100] * torch.tensor([1, -1, 1], device=device)

    obs = obs.view(obs.shape[0], num_frames * obs_dim)
    return obs


def _transform_critic_obs_left_right(env: BaseEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply left-right symmetry to the critic observation tensor.

    Critic observation layout (105 dims per frame):
        [0:100]   policy obs  (100)
        [100:103] lin_vel     (3)
        [103:105] feet_contact (2)
    """
    obs = obs.clone()
    device = obs.device
    obs_dim = 105
    assert obs.shape[1] % obs_dim == 0, \
        f"Obs dim {obs.shape[1]} is not divisible by {obs_dim}"
    num_frames = obs.shape[1] // obs_dim
    obs = obs.view(obs.shape[0], num_frames, obs_dim)

    # --- policy obs part (same as policy transform) ---
    obs[..., 0:3]   = obs[..., 0:3]   * torch.tensor([1, -1, -1], device=device)
    obs[..., 7:36]  = _switch_joints_left_right(obs[..., 7:36])
    obs[..., 36:65] = _switch_joints_left_right(obs[..., 36:65])
    obs[..., 65:94] = _switch_joints_left_right(obs[..., 65:94])
    obs[..., 94:97]  = obs[..., 94:97]  * torch.tensor([-1, 1, -1], device=device)
    obs[..., 97:100] = obs[..., 97:100] * torch.tensor([1, -1, 1], device=device)
    # --- critic-only extra ---
    obs[..., 100:103] = obs[..., 100:103] * torch.tensor([1, -1, 1], device=device)
    contact = obs[..., 103:105].clone()
    obs[..., 103:105] = contact[..., [1, 0]]  # swap left and right foot contact

    obs = obs.view(obs.shape[0], num_frames * obs_dim)
    return obs


# ============================================================
# Action transformation
# ============================================================

def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Apply left-right symmetry to the actions tensor."""
    actions = actions.clone()
    actions[:] = _switch_joints_left_right(actions[:])
    return actions


# ============================================================
# Joint-level left-right switch (29 DOF)
# ============================================================

"""
Unitree G1 29-DOF joint ordering (dof_names in unitree_env.py):

 0  left_hip_pitch_joint        Y  no-flip
 1  left_hip_roll_joint         X  flip
 2  left_hip_yaw_joint          Z  flip
 3  left_knee_joint             Y  no-flip
 4  left_ankle_pitch_joint      Y  no-flip
 5  left_ankle_roll_joint       X  flip
 6  right_hip_pitch_joint       Y  no-flip
 7  right_hip_roll_joint        X  flip
 8  right_hip_yaw_joint         Z  flip
 9  right_knee_joint            Y  no-flip
10  right_ankle_pitch_joint     Y  no-flip
11  right_ankle_roll_joint      X  flip
12  waist_yaw_joint             Z  flip
13  waist_roll_joint            X  flip
14  waist_pitch_joint           Y  no-flip
15  left_shoulder_pitch_joint   Y  no-flip
16  left_shoulder_roll_joint    X  flip
17  left_shoulder_yaw_joint     Z  flip
18  left_elbow_joint            Y  no-flip
19  left_wrist_roll_joint       X  flip
20  left_wrist_pitch_joint      Y  no-flip
21  left_wrist_yaw_joint        Z  flip
22  right_shoulder_pitch_joint  Y  no-flip
23  right_shoulder_roll_joint   X  flip
24  right_shoulder_yaw_joint    Z  flip
25  right_elbow_joint           Y  no-flip
26  right_wrist_roll_joint      X  flip
27  right_wrist_pitch_joint     Y  no-flip
28  right_wrist_yaw_joint       Z  flip

Left arm:  [15:22]  (7 joints: shoulder pitch/roll/yaw + elbow + wrist roll/pitch/yaw)
Right arm: [22:29]  (7 joints: same order)
"""


def _switch_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Apply left-right symmetry swap and sign corrections to joint data (29 DOF)."""
    joint_data_switched = joint_data.clone()

    # --- Swap left <-> right ---
    # Legs: [0:6] <-> [6:12]
    joint_data_switched[..., 0:6]   = joint_data[..., 6:12]
    joint_data_switched[..., 6:12]  = joint_data[..., 0:6]
    # Arms: [15:22] <-> [22:29]  (7 joints each, includes wrists)
    joint_data_switched[..., 15:22] = joint_data[..., 22:29]
    joint_data_switched[..., 22:29] = joint_data[..., 15:22]

    # --- Waist: yaw (Z) and roll (X) negate ---
    joint_data_switched[..., [12, 13]] = -1 * joint_data_switched[..., [12, 13]]

    # --- Roll (X-axis) and yaw (Z-axis) joints negate after swap ---
    # Indices after swap:
    #   legs:     1(hip_roll), 2(hip_yaw), 5(ankle_roll), 7(hip_roll), 8(hip_yaw), 11(ankle_roll)
    #   arms:     16(sh_roll), 17(sh_yaw), 19(wr_roll), 21(wr_yaw)   ← was right, now at left slots
    #             23(sh_roll), 24(sh_yaw), 26(wr_roll), 28(wr_yaw)   ← was left, now at right slots
    joint_data_switched[..., [1, 2, 5, 7, 8, 11, 16, 17, 19, 21, 23, 24, 26, 28]] = \
        -1 * joint_data_switched[..., [1, 2, 5, 7, 8, 11, 16, 17, 19, 21, 23, 24, 26, 28]]

    return joint_data_switched
