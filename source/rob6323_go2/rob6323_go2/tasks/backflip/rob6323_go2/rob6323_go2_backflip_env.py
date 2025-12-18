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

        # 1. Trajectory Targets
        target_pitch = -2 * math.pi * phase
        target_pitch_vel = -2 * math.pi / self.cfg.flip_period_s

        # 2. Orientation Reward (Conditional)
        _, current_pitch, _ = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        pitch_error = self._wrap_to_pi(current_pitch - target_pitch)
        
        # [CRITICAL CHANGE] Only care about orientation on the GROUND. 
        # In the air, we don't care about angle, only spin speed.
        # This prevents the "fear of falling" from stopping the flip.
        is_airborne = (phase >= self.cfg.airborne_phase_start) & (phase <= self.cfg.airborne_phase_end)
        orientation_mask = ~is_airborne 
        
        orientation_reward = orientation_mask * torch.exp(-(pitch_error**2) / (self.cfg.flip_pitch_sigma**2))

        # 3. [NEW] Aggressive Rate Reward (The "Kick")
        # Reward matching the spin speed (approx -4 rad/s)
        current_pitch_vel = self.robot.data.root_ang_vel_b[:, 1]
        rate_error = torch.abs(current_pitch_vel - target_pitch_vel)
        
        # Give a huge reward for spinning, ONLY during the flip phase
        rate_reward = is_airborne * torch.exp(-(rate_error**2) / (1.5**2))

        # 4. Takeoff (Upward Momentum)
        takeoff_mask = phase < self.cfg.takeoff_phase_portion
        vertical_velocity = torch.clamp(self.robot.data.root_lin_vel_w[:, 2], min=0.0)
        takeoff_reward = takeoff_mask * vertical_velocity

        # 5. Airborne (Clearance) - Keep this weak for now
        contact_forces = self._contact_sensor.data.net_forces_w_history[:, -1]
        contact_force_sum = torch.sum(torch.norm(contact_forces, dim=-1), dim=1)
        airborne_reward = is_airborne * torch.exp(-contact_force_sum / 100.0)

        # 6. Landing - Reward upright pose ONLY at the end
        landing_phase = phase > self.cfg.airborne_phase_end
        landing_pitch_error = torch.abs(self._wrap_to_pi(current_pitch))
        landing_reward = landing_phase * torch.exp(-(landing_pitch_error**2) / 0.5) # Looser sigma

        # 7. Smoothness
        action_delta = self._actions - self._previous_actions
        smoothness_penalty = torch.mean(torch.square(action_delta), dim=1)

        rewards = {
            # Scale orientation down so it doesn't dominate
            "backflip_orientation": orientation_reward * 1.0, 
            # Scale rate UP to force the spin
            "backflip_rate": rate_reward * 2.0,
            "backflip_takeoff": takeoff_reward * self.cfg.takeoff_vel_reward_scale,
            "backflip_airborne": airborne_reward * self.cfg.airborne_reward_scale,
            "backflip_landing": landing_reward * self.cfg.landing_reward_scale,
            "action_smoothness_penalty": -smoothness_penalty * self.cfg.action_smoothness_scale,
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
