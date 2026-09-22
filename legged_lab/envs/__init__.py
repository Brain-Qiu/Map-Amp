
from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from legged_lab.envs.unitree.unitree_env import UnitreeEnv

from legged_lab.envs.unitree.unitree_config_fsm import (
    UnitreeMapAMPAgentCfg,
    UnitreeMapAMPEnvCfg,
)
from legged_lab.utils.task_registry import task_registry
task_registry.register("unitree_map_amp", UnitreeEnv, UnitreeMapAMPEnvCfg(), UnitreeMapAMPAgentCfg())