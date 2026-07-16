# Legged Lab: Direct IsaacLab Workflow for Legged Robots

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-green.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-green)](https://github.com/isaac-sim/IsaacLab/tree/v2.3.0)
[![RSL_RL](https://img.shields.io/badge/RSL_RL-2.3.3-blue)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

## Overview

## Overview

This repository provides a workflow for training humanoid robots with multi-gait locomotion using IsaacLab. It introduces MAP-AMP, a unified policy framework that combines FSM-guided mode management, mode-separated adversarial motion priors, velocity curriculum learning, and symmetry regularization for robust and natural Stand, Walk, Run, and Transition behaviors.

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
# Option 1: HTTPS
git clone https://github.com/Brain-Qiu/Map_Amp.git

# Option 2: SSH
git clone git@github.com:Brain-Qiu/Map_Amp.git
```

### 1. Create conda environment

```bash
conda create -n map_amp python=3.11
conda activate map_amp

### 2. Install dependencies
```bash
pip install pip==24.3.1 setuptools==69.5.1 wheel==0.43.0 packaging==23.2
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install warp-lang==1.8.1

### 3. Install IsaacLab
```bash
cd IsaacLab
./isaaclab.sh --install

### 4. Setup Isaac Sim
```bash
source /path/to/isaacsim/setup_conda_env.sh

### 5. Install MAP-AMP
```bash
cd map_amp/legged_lab/rsl_rl
pip install -e .
cd ../..
pip install -e . --no-deps

- Verify that the extension is correctly installed by running the following command:
```bash
python legged_lab/scripts/train.py --task=unitree_map_amp --headless --logger=tensorboard --num_envs=64
```

## Usage

### Motion Retargeting

**1. Prepare the dataset and Motion retargeting with [GMR](https://github.com/YanjieZe/GMR).**
```bash
python scripts/smplx_to_robot.py --smplx_file <path_to_smplx_data> --robot tienkung  --save_path <path_to_save_robot_data.pkl>
```
**2. Data Processing and Data Saving.**

The dataset consists of two parts with distinct functions and formats, requiring conversion in two steps.

- **`motion_visualization/`**  
  Used for motion playback with `play_amp_animation.py` to check motion correctness and quality.  
  Data fields:  [root_pos, root_rot, dof_pos, root_lin_vel, root_ang_vel, dof_vel]

- **`motion_amp_expert/`**  
  Used during training as expert reference data for AMP.  
  Data fields:  [dof_pos, dof_vel, end-effector pos]
  
- **Step 1: Data Processing and Visualization Data Saving.**

```bash
python legged_lab/scripts/gmr_data_conversion.py --input_pkl <path_to_save_robot_data.pkl> --output_txt legged_lab/envs/t1pro/datasets/motion_visualization/motion.txt
```

**Note**: Before starting step 2, set the `amp_motion_files_display` path in the config to the file generated in step 1.

- **Step 2: Motion Visualization and Expert Data Saving.**
```bash
python legged_lab/scripts/play_amp_animation.py --task=unitree_map_amp --save_path legged_lab/envs/t1pro/datasets/motion_amp_expert/motion.txt
```
**Note**: After step 2, set the `amp_motion_files` path in the config to the file generated in step 2.

### Visualize motion

Visualize the motion by updating the simulation with data from tienkung/datasets/motion_visualization.

```bash
python legged_lab/scripts/play_amp_animation.py --task=unitree_map_amp
```

### Train

Train the policy using AMP expert data from t1pro/datasets/motion_amp_expert.

```bash
python legged_lab/scripts/train.py --task=unitree_map_amp --headless --logger=tensorboard --num_envs=4096
```

Train the policy using RL.

```bash
python legged_lab/scripts/train.py --task=unitree_map_amp --headless --logger=tensorboard --num_envs=4096
```
### Play

Run the trained policy.

```bash
python legged_lab/scripts/play_unitree_fsm.py --task=unitree_map_amp   #map_amp task
```

### Sim2Sim(MuJoCo)

Evaluate the trained policy in MuJoCo to perform cross-simulation validation.

Exported_policy/ contains pretrained policies provided by the project. When using the play script, trained policy is exported automatically and saved to path like logs/[experiment_name]/[timestamp]/exported/policy.pt.(remerber out of the conda environment,like in base env)
```bash
python legged_lab/scripts/sim2sim_unitree_fsm.py --experiment_name unitree_map_amp --load_run *-*-*_*-*-*
```
```

## References and Thanks
This project repository builds upon the shoulders of giants.
* [IsaacLab](https://github.com/isaac-sim/IsaacLab)   The various reusable practical components in IsaacLab greatly simplify the complexity of LeggedLab.
* [legged_gym](https://github.com/leggedrobotics/legged_gym)   We borrowed the code organization and environment definition logic of legged_gym and simplified it as much as possible.
* [Protomotions](https://github.com/NVlabs/ProtoMotions)   The motivation for building this repository comes from protomotions. For the first time, we realized that we could create our own environment using only IsaacLab components without inheriting 'DirectRLEnv' or 'ManagerBasedRLEnv'.

## Citation

If you use Legged Lab in your research, you can cite it as follows:

```bibtex
@software{LeggedLab,
  author = {Tlibot, Qiu},
  license = {BSD-3-Clause},
  title = {Legged Lab: amp and rl Workflow for Legged Robots},
  url = {https://github.com/Brain-Qiu/LeggedLab_Amp},
  version = {1.0.0},
  year = {2026}
}
```
