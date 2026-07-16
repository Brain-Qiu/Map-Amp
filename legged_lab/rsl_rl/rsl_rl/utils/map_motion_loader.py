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

import glob
import os
from rsl_rl.utils.motion_loader import AMPLoader

class MapAMPLoaderGroup:
    """
    A container that physically isolates motion datasets into N independent AMPLoaders
    based on the file name prefix and the provided `mode_index_map`.
    """
    
    def __init__(
        self,
        device,
        time_between_frames,
        preload_transitions=False,
        num_preload_transitions=1000000,
        motion_files=[],
        mode_index_map=None,
        joint_pos_size=20,
        joint_vel_size=20,
        end_effector_pos_size=12,
        num_disc_frames=2,
    ):
        self.device = device
        self.num_modes = len(mode_index_map) if mode_index_map else 1
        self.mode_index_map = mode_index_map
        self.num_disc_frames = num_disc_frames

        # Parse and route files
        routed_files_by_mode = {m: [] for m in range(self.num_modes)}
        
        for file_path in motion_files:
            file_name = os.path.basename(file_path).lower()
            assigned_mode = None
            
            for mode_str, mode_id in self.mode_index_map.items():
                # We assume mode_str like 'run', 'walk' is in the filename.
                if mode_str.lower() in file_name:
                    assigned_mode = mode_id
                    break
            
            if assigned_mode is not None:
                routed_files_by_mode[assigned_mode].append(file_path)
            else:
                print(f"Warning: MapAMPLoader could not route file {file_name} to any expert pool.")

        # Initialize isolated AMPLoaders
        self.loaders = []
        for i in range(self.num_modes):
            mode_files = routed_files_by_mode[i]
            if len(mode_files) > 0:
                print(f"[MapAMPLoader] Initializing Loader for mode {i} with {len(mode_files)} files...")
                loader = AMPLoader(
                    device=device,
                    time_between_frames=time_between_frames,
                    preload_transitions=preload_transitions,
                    # We proportionally divide preload budget
                    num_preload_transitions=max(1, num_preload_transitions // self.num_modes),
                    motion_files=mode_files,
                    joint_pos_size=joint_pos_size,
                    joint_vel_size=joint_vel_size,
                    end_effector_pos_size=end_effector_pos_size,
                    num_disc_frames=num_disc_frames,
                )
                self.loaders.append(loader)
            else:
                print(f"[MapAMPLoader] WARNING: No dataset found for mode {i}. Loader will be None.")
                self.loaders.append(None)

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """
        Generates a batch of AMP transitions synchronously for all available modes.
        Yields a dict mapping mode_id to a tuple of num_disc_frames tensors.
        """
        # Create underlying generators
        generators = []
        for i in range(self.num_modes):
            if self.loaders[i] is not None:
                generators.append(self.loaders[i].feed_forward_generator(num_mini_batch, mini_batch_size))
            else:
                generators.append(None)
                
        for _ in range(num_mini_batch):
            samples_dict = {}
            for i in range(self.num_modes):
                if generators[i] is not None:
                    samples_dict[i] = next(generators[i])  # N-tuple of tensors
                else:
                    samples_dict[i] = None
            yield samples_dict
