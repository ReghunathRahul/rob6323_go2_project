
from isaaclab.utils import configclass
from rob6323_go2.tasks.direct.rob6323_go2.rob6323_go2_env_cfg import Rob6323Go2EnvCfg
import torch

@configclass
class Rob6323Go2BackflipEnvCfg(Rob6323Go2EnvCfg):
    """
    Backflip specific tuning 
    """

    episode_length_s = 1.2
    flip_period_s = 0.9

    action_scale = 1.0
    action_smoothness_scale = 0.05 
    debug_vis = False

    curriculum_duration_steps: float = 5_000_000.0

    observation_space = Rob6323Go2EnvCfg().observation_space + 2

    takeoff_phase_portion: float = 0.25
    airborne_phase_start: float = 0.25
    airborne_phase_end: float = 0.75
    flip_pitch_sigma: float = 0.5
    air_contact_force_limit: float = 120.0
    landing_pitch_sigma: float = 0.25
    landing_force_sigma: float = 120.0
    orientation_reward_scale: float = 1.5
    takeoff_vel_reward_scale: float = 0.6
    airborne_reward_scale: float = 0.4
    landing_reward_scale: float = 0.7
    target_flip_speed: float = -8.0

    scene = Rob6323Go2EnvCfg().scene.replace(num_envs=2048, env_spacing=5.0)
    scene.terrain.static_friction = 0.7
    scene.terrain.dynamic_friction = 0.6
    contact_sensor = Rob6323Go2EnvCfg().contact_sensor.replace(history_length=5)

    def __post_init__(self):
        super().__post_init__()

        if self.actuators is not None:
             for key in self.actuators.keys():
                self.actuators[key].stiffness = 60.0
                self.actuators[key].damping = 5.0 

        self.initial_state.pos = (0.0, 0.0, 0.55)
        self.initial_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.initial_state.joint_pos = {
            ".*hip_joint": 0.0,
            ".*thigh_joint": 0.8,
            ".*calf_joint": -1.5,
        }
        self.initial_state.joint_vel = {".*": 0.0}