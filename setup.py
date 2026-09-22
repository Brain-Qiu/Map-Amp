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

from distutils.core import setup

from setuptools import find_packages

setup(
    name="MAP_AMP",
    packages=find_packages(),
    version="1.0.0",
    install_requires=[
    "h5py==3.16.0",
    "tensordict==0.13.0",
    "tensorboard==2.21.0",

    "etils==1.14.0",
    "glfw==2.10.2",
    "mujoco==3.13.0",
    "PyOpenGL==3.1.10",

    "evdev==2.0.0",
    "pynput==1.8.2",
    "python-xlib==0.33",
    "six==1.17.0",

    "scipy==1.17.1",
    "PyYAML==6.0.3",

    # "isaacsim",
    # "IsaacLab==5.1.0",
    # "rsl-rl-lib==2.3.0",
],
)

