import pickle
import numpy as np
import torch
import argparse
import os
import glob
from isaaclab.utils.math import quat_mul, quat_conjugate, axis_angle_from_quat  
from scipy.spatial.transform import Rotation 

def convert_pkl_to_custom(input_pkl, output_txt, fps, input_fps=None):
    if input_fps is None:
        input_fps = fps
    dt = 1.0 / fps
    # 兼容不同NumPy版本
    try:
        with open(input_pkl, "rb") as f:
            motion_data = pickle.load(f)
    except ModuleNotFoundError as e:
        print(f"⚠️  NumPy版本兼容问题: {e}")
        print("尝试使用兼容模式加载...")
        with open(input_pkl, "rb") as f:
            motion_data = pickle.load(f, encoding='latin1')

    #gmr数据转换，70维顺序：root_pos（3）, root_euler（4）, dof_pos（29）, 
    #root_lin_vel（3）, root_ang_vel（3）, dof_vel（29）
    root_pos = motion_data["root_pos"]     
    root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]]  # xyzw → wxyz
    dof_pos = motion_data["dof_pos"]

    # Downsample if input_fps differs from output fps
    stride = int(round(input_fps / fps))
    if stride > 1:
        root_pos = root_pos[::stride]
        root_rot = root_rot[::stride]
        dof_pos = dof_pos[::stride]

    root_lin_vel = (root_pos[1:] - root_pos[:-1]) / dt
    root_rot_t = torch.tensor(root_rot, dtype=torch.float32)

    q1_conj = quat_conjugate(root_rot_t[:-1])         
    dq = quat_mul(q1_conj, root_rot_t[1:])            
    axis_angle = axis_angle_from_quat(dq)             
    root_ang_vel = axis_angle / dt

    dof_vel = (dof_pos[1:] - dof_pos[:-1]) / dt

    euler_angles = Rotation.from_quat(root_rot[:-1, [1, 2, 3, 0]]).as_euler('XYZ', degrees=False)
    euler_angles = np.unwrap(euler_angles, axis=0)

    data_output = np.concatenate(
        (root_pos[:-1], euler_angles, dof_pos[:-1],  
         root_lin_vel, root_ang_vel, dof_vel),
        axis=1
    )

    np.savetxt(output_txt, data_output, fmt='%f', delimiter=', ')
    with open(output_txt, 'r') as f:
        frames_data = f.readlines()

    frames_data_len = len(frames_data)
    with open(output_txt, 'w') as f:
        f.write('{\n')
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {1.0/fps:.3f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write('"MotionWeight": 0.5,\n\n')
        f.write('"Frames":\n[\n')

        for i, line in enumerate(frames_data):
            line_start_str = '  ['
            if i == frames_data_len - 1:
                f.write(line_start_str + line.rstrip() + ']\n')
            else:
                f.write(line_start_str + line.rstrip() + '],\n')

        f.write(']\n}')
    print(f"✅ Successfully converted {input_pkl} to {output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, default=None)
    parser.add_argument("--output_txt", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None, help="Batch mode: input folder containing .pkl files")
    parser.add_argument("--output_dir", type=str, default=None, help="Batch mode: output folder for converted .txt files")
    parser.add_argument("--fps", type=float, default=30.0, help="Output frame rate")
    parser.add_argument("--input_fps", type=float, default=None, help="Input frame rate (default: same as --fps, no downsampling)")
    args = parser.parse_args()

    if args.input_dir is not None:
        # Batch folder conversion
        if args.output_dir is None:
            parser.error("--output_dir is required when using --input_dir")
        pkl_files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.pkl"), recursive=True))
        if not pkl_files:
            print(f"No .pkl files found in {args.input_dir}")
        for input_pkl in pkl_files:
            rel_path = os.path.relpath(input_pkl, args.input_dir)
            output_txt = os.path.join(args.output_dir, os.path.splitext(rel_path)[0] + ".txt")
            os.makedirs(os.path.dirname(output_txt), exist_ok=True)
            convert_pkl_to_custom(input_pkl, output_txt, args.fps, args.input_fps)
    else:
        # Single file conversion
        if args.input_pkl is None or args.output_txt is None:
            parser.error("--input_pkl and --output_txt are required for single file mode")
        convert_pkl_to_custom(args.input_pkl, args.output_txt, args.fps, args.input_fps)