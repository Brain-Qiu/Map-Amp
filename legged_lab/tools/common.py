# Copyright (c) 2022-2024, fudeRobot All rights reserved.
#
#

import os
import yaml

ENV_CFG_FILENAME = '01_env_cfg.yaml'
TRAIN_CFG_FILENAME = '01_train_cfg.yaml'
TRAINCOPY_ASSET_FILENAME = '01_urdf.urdf'
TRAINCOPY_MJCF_FILENAME = '01_mjcf.xml'
TRAINCOPY_SIM2SIMCFG_FILENAME = '01_sim2sim_cfg.yaml'
SIM2SIMCFG_FILENAME = 'sim2sim_cfg.yaml'
TRAINCOPY_ASSET_DIR = 'train_asset'


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type) or type(attr).__module__ != 'builtins':
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def save_class_to_yaml_file(obj, filename):
    dct = class_to_dict(obj)
    if len(dct) == 0:
        return
    dctyaml = yaml.dump(dct)

    # 提取文件夹路径
    folder = os.path.dirname(filename)
    # 如果文件夹不存在，就创建它
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(filename, 'w') as f:
        f.write(dctyaml)


def update_class_from_yaml_file(obj, filename):
    with open(filename, 'r') as yaml_file:
        data = yaml.load(yaml_file, Loader=yaml.FullLoader)
        update_class_from_dict(obj, data)


def get_asset_path_train_backup(src_asset_file):
    asset_common_path = os.path.dirname(os.path.dirname(src_asset_file))
    target_path = os.path.join(asset_common_path, TRAINCOPY_ASSET_DIR)
    return target_path


def get_mjcf_filename_train_backup(src_asset_file, load_run : str):
    """Get the filename of mjcf to backup, when training.

    Args:
        src_asset_file (_type_): path for urdf
        load_run (str): name of the load_run

    Returns:
        _type_: _description_
    """
    target_path = get_asset_path_train_backup(src_asset_file)
    filename = os.path.basename(src_asset_file)
    name, ext = os.path.splitext(filename)
    dist_mjcf_file = os.path.join(target_path, f'{name}_{load_run}.xml')
    return dist_mjcf_file
