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
        current_step = self.common_step_counter
        
        
        # Shorten to 3M steps to see results faster
        curriculum_duration = 3_000_000.0 
        curriculum_factor = torch.clamp(torch.tensor(current_step / curriculum_duration), 0.0, 1.0).to(self.device)
        
        # Target spin ramps from -6.0 to -12.0 rad/s (Need FAST spin for backflip)
        target_spin_speed = -6.0 + (curriculum_factor * -6.0) 

        # strict phases to prevent "blurring"
        is_takeoff = phase < 0.2
        is_airborne = (phase >= 0.2) & (phase <= 0.75)
        is_landing = phase > 0.75
        
        root_lin_vel = self.robot.data.root_lin_vel_w
        root_ang_vel = self.robot.data.root_ang_vel_b
        root_quat = self.robot.data.root_quat_w
        _, current_pitch, _ = math_utils.euler_xyz_from_quat(root_quat)

        # Reward upward velocity
        vel_z = torch.clamp(root_lin_vel[:, 2], min=0.0)
        takeoff_reward = is_takeoff * vel_z * 2.0
        
        # Penalize rotation during takeoff (prevent slipping)
        takeoff_stable_reward = is_takeoff * torch.exp(-(root_ang_vel[:, 1]**2) / 1.0)

        # Orientation reward is ZERO here. 
        # We ONLY reward spinning pitch velocity.
        current_spin = root_ang_vel[:, 1]
        
        # Rate reward: Gaussian focused on the target spin
        spin_error = current_spin - target_spin_speed
        # We use a wider sigma initially so it finds the gradient easily
        rate_reward = is_airborne * torch.exp(-(spin_error**2) / 5.0)
        
        # Optional: Penalize touching the ground in the air (drag)
        contact_forces = torch.norm(self._contact_sensor.data.net_forces_w_history[:, -1], dim=-1).sum(dim=1)
        air_penalty = is_airborne * (contact_forces > 1.0).float() * -0.5

        # Now we turn Orientation reward BACK ON.
        # Target is effectively 0 pitch (upright)
        pitch_error = torch.abs(self._wrap_to_pi(current_pitch))
        land_orient_reward = is_landing * torch.exp(-(pitch_error**2) / 0.5)
        
        # Reward low velocity (stability)
        vel_xy = torch.norm(root_lin_vel[:, :2], dim=-1)
        land_still_reward = is_landing * torch.exp(-vel_xy / 1.0)

        action_smoothness = -torch.mean(torch.square(self._actions - self._previous_actions), dim=1)

        rewards = {
            "takeoff_power": takeoff_reward * 1.0,
            "takeoff_stable": takeoff_stable_reward * 0.5,
            
            # Massive weight on spin to force the flip
            "air_spin": rate_reward * 4.0, 
            "air_penalty": air_penalty,
            
            "land_orient": land_orient_reward * 1.0,
            "land_still": land_still_reward * 0.5,
            
            "smoothness": action_smoothness * 0.05
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
