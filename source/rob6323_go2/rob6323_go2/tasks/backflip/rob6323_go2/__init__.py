# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from .rob6323_go2_backflip_env import Rob6323Go2BackflipEnv
from .rob6323_go2_backflip_env_cfg import Rob6323Go2BackflipEnvCfg
from .rob6323_go2_jump_env import Rob6323Go2jumpEnv
from .rob6323_go2_jump_env_cfg import Rob6323Go2jumpEnvCfg

__all__ = [
    "rob6323go2backflipenv",
    "rob6323go2backflipenvcfg",
    "rob6323go2jumpenv",
    "rob6323go2jumpenvcfg",
]

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-Rob6323-Go2-Backflip-v0",
    entry_point=f"{__name__}.rob6323_go2_backflip_env:Rob6323Go2BackflipEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rob6323_go2_backflip_env_cfg:Rob6323Go2BackflipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
gym.register(
    id="Template-Rob6323-Go2-Jump-v0",
    entry_point=f"{__name__}.rob6323_go2_jump_env:Rob6323Go2jumpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rob6323_go2_jump_env_cfg:Rob6323Go2jumpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
