# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.actuators import ImplicitActuatorCfg

from rob6323_go2.tasks.direct.rob6323_go2.rob6323_go2_env_cfg import Rob6323Go2EnvCfg


@configclass
class Rob6323Go2BackflipEnvCfg(Rob6323Go2EnvCfg):
    # env
    decimation = 4
    episode_length_s = 1.2
    flip_period_s = 0.9

    # spaces definition
    action_scale = 1.0
    action_space = 12
    observation_space = 48 + 2
    state_space = 0
    debug_vis = False

    # PD controller
    Kp = 60.0
    Kd = 5.0
    torque_limits = 100.0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.7,
            dynamic_friction=0.6,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot(s)
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit_sim=23.5,
        velocity_limit_sim=30.0,
        stiffness=Kp,
        damping=Kd,
    )

    robot_cfg.init_state.pos = (0.0, 0.0, 0.55)
    robot_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    robot_cfg.init_state.joint_pos = {
        ".*hip_joint": 0.0,
        ".*thigh_joint": 0.8,
        ".*calf_joint": -1.5,
    }
    robot_cfg.init_state.joint_vel = {".*": 0.0}

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=4.0, replicate_physics=True)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=5, update_period=0.005, track_air_time=True
    )

    # reward scales
    curriculum_duration_steps = 5_000_000.0
    
    takeoff_phase_portion = 0.25
    airborne_phase_start = 0.25
    airborne_phase_end = 0.75
    flip_pitch_sigma = 0.5
    air_contact_force_limit = 120.0
    landing_pitch_sigma = 0.25
    landing_force_sigma = 120.0
    
    orientation_reward_scale = 1.5
    takeoff_vel_reward_scale = 0.6
    airborne_reward_scale = 0.4
    landing_reward_scale = 0.7
    target_flip_speed = -8.0
    action_smoothness_scale = 0.05

    # actuator friction randomization
    viscous_friction_range = (0.0, 0.3)
    stiction_friction_range = (0.0, 2.5)
    stiction_vel_tol = 0.1