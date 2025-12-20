from isaaclab.utils import configclass
from rob6323_go2.tasks.direct.rob6323_go2.rob6323_go2_env_cfg import Rob6323Go2EnvCfg

@configclass
class Rob6323Go2BackflipEnvCfg(Rob6323Go2EnvCfg):
    """
    Vertical Hopping Tuning (The "Elegant Jump")
    """
   
    episode_length_s = 1.5
    observation_space = Rob6323Go2EnvCfg().observation_space + 2
    action_scale = 0.5 

    
    flip_period_s: float = 1.5   # Matches the rhythm in your video
    takeoff_phase_portion: float = 0.2
    airborne_phase_start: float = 0.2
    airborne_phase_end: float = 0.75
    
    
    takeoff_vel_reward_scale: float = 3.0  
    landing_reward_scale: float = 2.0        
    
    # Penalties & Regularization
    penalty_torque_scale: float = -1.0e-4 
    penalty_lin_vel_xy_scale: float = -2.0   
    penalty_ang_vel_xy_scale: float = -0.5   
    penalty_pitch_instability_scale: float = -1.5 
    
    target_jump_vel: float = 2.5    
    target_flip_speed: float = 0.0  

   
    scene = Rob6323Go2EnvCfg().scene.replace(num_envs=2048, env_spacing=5.0)
    contact_sensor = Rob6323Go2EnvCfg().contact_sensor.replace(history_length=5)