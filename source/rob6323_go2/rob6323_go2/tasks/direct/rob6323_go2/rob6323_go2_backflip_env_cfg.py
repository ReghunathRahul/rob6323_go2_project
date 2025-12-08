"""
This configuration adapts the horizon, observation space, and reward weights for a backflip cycle.
"""

from isaaclab.utils import configclass

from .rob6323_go2_env_cfg import Rob6323Go2EnvCfg


@configclass
class Rob6323Go2BackflipEnvCfg(Rob6323Go2EnvCfg):
    """
    Backflip-specific tuning
    """

    # shorter episodes for a clean flip and landing rather than locomotion.
    episode_length_s = 8.0

    # add features (sin/cos) to help the policy track where it is in flip.
    observation_space = Rob6323Go2EnvCfg.observation_space + 2

    action_scale = 0.3
    debug_vis = False

    # task timing and reward shaping parameters
    flip_period_s: float = 1.6
    takeoff_phase_portion: float = 0.25
    airborne_phase_start: float = 0.25
    airborne_phase_end: float = 0.75
    flip_pitch_sigma: float = 0.35
    air_contact_force_limit: float = 120.0
    landing_pitch_sigma: float = 0.25
    landing_force_sigma: float = 120.0
    action_smoothness_scale: float = 0.05
    orientation_reward_scale: float = 1.0
    takeoff_vel_reward_scale: float = 0.6
    airborne_reward_scale: float = 0.4
    landing_reward_scale: float = 0.7

    # environment setup
    scene = Rob6323Go2EnvCfg.scene.replace(num_envs=2048, env_spacing=5.0)
    contact_sensor = Rob6323Go2EnvCfg.contact_sensor.replace(history_length=5)
