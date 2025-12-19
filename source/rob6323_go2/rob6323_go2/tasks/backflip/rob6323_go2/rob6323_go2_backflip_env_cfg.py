# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.actuators import ImplicitActuatorCfg

from rob6323_go2.tasks.direct.rob6323_go2.rob6323_go2_env_cfg import Rob6323Go2EnvCfg

@configclass
class Rob6323Go2BackflipEnvCfg(Rob6323Go2EnvCfg):
 
    episode_length_s = 6.0
    
    flip_period_s = 5.0

    action_scale = 1.0
    
    Kp = 20.0
    Kd = 0.5
    torque_limits = 45.0 

    observation_space = 48 + 4 + 2 

    curriculum_duration_steps = 5_000_000.0
    
    # Phases
    takeoff_phase_portion = 0.2
    airborne_phase_start = 0.5
    airborne_phase_end = 0.8
    
    # Weights (Aggressive)
    lin_vel_z_reward_scale = 25.0      
    ang_vel_y_reward_scale = 10.0       
    orientation_reward_scale = 5.0      
    
    raibert_heuristic_reward_scale = 0.0
    feet_clarety_reward_scale = 0.0
    
    # New Backflip specific scales
    rear_up_reward_scale = 5.0
    air_tuck_reward_scale = 1.0