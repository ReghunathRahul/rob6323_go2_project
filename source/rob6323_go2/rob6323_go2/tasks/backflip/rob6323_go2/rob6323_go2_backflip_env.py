"""
Backflips!
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
        
        # --- 1. Compute Curriculum Factor ---
        # 0.0 = Start of training (Easy)
        # 1.0 = End of curriculum (Hard)
        # Note: self.common_step_counter is standard in Isaac Lab. 
        # If it throws an error, use self.episode_length_buf.mean() or similar, 
        # but usually common_step_counter is available in DirectRLEnv.
        current_step = self.common_step_counter
        curriculum_factor = torch.clamp(
            torch.tensor(current_step / self.cfg.curriculum_duration_steps), 
            min=0.0, max=1.0
        ).to(self.device)

        # --- 2. Define Phases ---
        is_takeoff = phase < self.cfg.takeoff_phase_portion
        is_airborne = (phase >= self.cfg.airborne_phase_start) & (phase <= self.cfg.airborne_phase_end)
        is_landing = phase > self.cfg.airborne_phase_end
        
        # State
        root_lin_vel = self.robot.data.root_lin_vel_w
        root_ang_vel = self.robot.data.root_ang_vel_b
        _, current_pitch, _ = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)

        # --- 3. REWARD: Takeoff ---
        vertical_vel = torch.clamp(root_lin_vel[:, 2], min=0.0)
        takeoff_reward = is_takeoff * vertical_vel * 1.5 
        
        # --- 4. REWARD: Airborne (Curriculum Applied Here!) ---
        # Start target at 0.0, ramp down to -8.0 (or whatever is in cfg)
        target_spin = self.cfg.target_flip_speed * curriculum_factor
        current_spin = root_ang_vel[:, 1]
        
        # Calculate error
        spin_error = torch.abs(current_spin - target_spin)
        
        # As the task gets harder (target_spin increases), we can loosen the sigma slightly
        # to prevent frustration, or keep it constant.
        rate_reward = is_airborne * torch.exp(-(spin_error**2) / (4.0**2))

        # --- 5. REWARD: Landing ---
        landing_pitch_error = torch.abs(self._wrap_to_pi(current_pitch))
        landing_reward = is_landing * torch.exp(-(landing_pitch_error**2) / 0.5)
        
        # Landing Stability (Stop moving horizontally)
        landing_vel_penalty = is_landing * torch.norm(root_lin_vel[:, :2], dim=-1)
        landing_reward *= torch.exp(-landing_vel_penalty / 1.0)

        # --- 6. Penalties ---
        action_delta = self._actions - self._previous_actions
        smoothness_penalty = torch.mean(torch.square(action_delta), dim=1)

        # --- 7. Compose ---
        rewards = {
            "takeoff_impulse": takeoff_reward * self.cfg.takeoff_vel_reward_scale,
            "backflip_spin": rate_reward * 3.0, 
            "landing_stability": landing_reward * self.cfg.landing_reward_scale,
            "action_smoothness": -smoothness_penalty * self.cfg.action_smoothness_scale,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
             if key not in self._episode_sums:
                self._episode_sums[key] = torch.zeros_like(value)
             self._episode_sums[key] += value
             
        return reward

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
