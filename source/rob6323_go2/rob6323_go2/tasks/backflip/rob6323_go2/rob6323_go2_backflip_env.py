
"""
Backflips
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils

from rob6323_go2.tasks.direct.rob6323_go2.rob6323_go2_env import Rob6323Go2Env

from .rob6323_go2_backflip_env_cfg import Rob6323Go2BackflipEnvCfg


class Rob6323Go2BackflipEnv(Rob6323Go2Env):
    """
    Task for training periodic backflips.
    """

    cfg: Rob6323Go2BackflipEnvCfg

    def __init__(self, cfg: Rob6323Go2BackflipEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._episode_sums.update(
            {
                "backflip_orientation": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
                "backflip_takeoff": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
                "backflip_airborne": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
                "backflip_landing": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
                "action_smoothness_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            }
        )

    def _get_observations(self,
    ) -> dict:
        observations = super()._get_observations()
        phase = self._compute_phase()
        phase_features = torch.stack([torch.sin(2 * math.pi * phase), torch.cos(2 * math.pi * phase)], dim=1)
        observations["policy"] = torch.cat([observations["policy"], phase_features], dim=-1)
        return observations

    def _get_rewards(self) -> torch.Tensor:
        phase = self._compute_phase()
        
    
        root_lin_vel = self.robot.data.root_lin_vel_w
        root_ang_vel = self.robot.data.root_ang_vel_b
        _, current_pitch, _ = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)

     
        is_takeoff = phase < self.cfg.takeoff_phase_portion
        is_airborne = (phase >= self.cfg.airborne_phase_start) & (phase <= self.cfg.airborne_phase_end)
        is_landing = phase > self.cfg.airborne_phase_end

      
        current_vel_z = root_lin_vel[:, 2]
        vel_error = torch.abs(current_vel_z - self.cfg.target_jump_vel)
        r_takeoff = is_takeoff * torch.exp(-torch.square(vel_error) / 0.5)

        
        pitch_error = torch.abs(self._wrap_to_pi(current_pitch))
        
        r_pitch_stability = is_airborne * torch.exp(-torch.square(pitch_error) / 0.1)

        
        landing_vel_xy = torch.norm(root_lin_vel[:, :2], dim=-1)
        landing_vel_z = torch.abs(root_lin_vel[:, 2])
        
        r_landing = is_landing * (
            torch.exp(-torch.square(pitch_error) / 0.2) * 
            torch.exp(-torch.square(landing_vel_xy) / 0.5) * 
            torch.exp(-torch.square(landing_vel_z) / 1.0)    
        )

    
        torques = self.robot.data.applied_torque
        p_torque = torch.sum(torch.square(torques), dim=1)

        p_lin_vel_xy = torch.sum(torch.square(root_lin_vel[:, :2]), dim=1) 
        
        p_pitch_instability = torch.square(root_ang_vel[:, 1]) 

        rewards = {
            "reward_takeoff": r_takeoff * self.cfg.takeoff_vel_reward_scale,
            "reward_landing": r_landing * self.cfg.landing_reward_scale,
            "reward_stability": r_pitch_stability * 1.0, 
            
            "penalty_torque": p_torque * self.cfg.penalty_torque_scale,
            "penalty_lin_vel_xy": p_lin_vel_xy * self.cfg.penalty_lin_vel_xy_scale,
            "penalty_pitch_instability": p_pitch_instability * self.cfg.penalty_pitch_instability_scale,
        }

        return torch.sum(torch.stack(list(rewards.values())), dim=0)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        self._commands[env_ids] = 0.0

    def _compute_phase(self,
    ) -> torch.Tensor:
        """
        Compute the normalized phase [0, 1) within the flip period.
        """
        phase = (self.progress_buf * self.step_dt) / self.cfg.flip_period_s
        return phase % 1.

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor
    ) -> torch.Tensor:
        """
        Wrap raw angles to the [-pi, pi] range for stable error metrics
        """
        return torch.atan2(torch.sin(angle), torch.cos(angle))
