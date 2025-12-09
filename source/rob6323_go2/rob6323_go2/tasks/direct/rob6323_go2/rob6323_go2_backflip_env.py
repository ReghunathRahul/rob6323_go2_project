"""
Backflips!
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils

from .rob6323_go2_backflip_env_cfg import Rob6323Go2BackflipEnvCfg
from .rob6323_go2_env import Rob6323Go2Env


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

    def _get_rewards(self,
    ) -> torch.Tensor:
        phase = self._compute_phase()
        target_pitch = (phase * 2 * math.pi) - math.pi

        # stay close to the pitch target
        _, current_pitch, _ = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        pitch_error = self._wrap_to_pi(current_pitch - target_pitch)
        orientation_reward = torch.exp(-(pitch_error**2) / (self.cfg.flip_pitch_sigma**2))

        # reward upward momentum during the takeoff
        takeoff_mask = phase < self.cfg.takeoff_phase_portion
        vertical_velocity = torch.clamp(self.robot.data.root_lin_vel_b[:, 2], min=0.0)
        takeoff_reward = takeoff_mask * vertical_velocity

        # use contact forces to encourage being airborne during the mid
        contact_forces = self._contact_sensor.data.net_forces_w_history[:, -1]
        contact_force_sum = torch.sum(torch.norm(contact_forces, dim=-1), dim=1)
        airborne_phase = (phase >= self.cfg.airborne_phase_start) & (phase <= self.cfg.airborne_phase_end)
        airborne_reward = airborne_phase * torch.exp(-contact_force_sum / (self.cfg.air_contact_force_limit + 1e-6))

        # reward landing upright with low forces
        landing_phase = phase > self.cfg.airborne_phase_end
        landing_pitch_error = torch.abs(self._wrap_to_pi(current_pitch))
        landing_reward = landing_phase * torch.exp(-(landing_pitch_error**2) / (self.cfg.landing_pitch_sigma**2))
        landing_reward *= torch.exp(-contact_force_sum / (self.cfg.landing_force_sigma + 1e-6))

        # penalize aggressive action changes to keep flips controllable
        action_delta = self._actions - self._previous_actions
        smoothness_penalty = torch.mean(torch.square(action_delta), dim=1)

        rewards = {
            "backflip_orientation": orientation_reward * self.cfg.orientation_reward_scale,
            "backflip_takeoff": takeoff_reward * self.cfg.takeoff_vel_reward_scale,
            "backflip_airborne": airborne_reward * self.cfg.airborne_reward_scale,
            "backflip_landing": landing_reward * self.cfg.landing_reward_scale,
            "action_smoothness_penalty": -smoothness_penalty * self.cfg.action_smoothness_scale,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
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
