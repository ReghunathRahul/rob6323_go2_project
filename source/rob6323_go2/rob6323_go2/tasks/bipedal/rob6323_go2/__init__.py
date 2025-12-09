# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from .rob6323_go2_bipedal_env import Rob6323Go2BipedalEnv
from .rob6323_go2_bipedal_env_cfg import Rob6323Go2BipedalEnvCfg

__all__ = [
    "Rob6323Go2BipedalEnv",
    "Rob6323Go2BipedalEnvCfg",
]

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-Rob6323-Go2-Bipedal-v0",
    entry_point=f"{__name__}.rob6323_go2_bipedal_env:Rob6323Go2BipedalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rob6323_go2_bipedal_env_cfg:Rob6323Go2BipedalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
