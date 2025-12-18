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
        
        # --- 1. Define Phases ---
        # Takeoff: 0.0 -> 0.25
        # Air:     0.25 -> 0.75
        # Landing: 0.75 -> 1.0
        is_takeoff = phase < self.cfg.takeoff_phase_portion
        is_airborne = (phase >= self.cfg.airborne_phase_start) & (phase <= self.cfg.airborne_phase_end)
        is_landing = phase > self.cfg.airborne_phase_end
        
        # --- 2. Get State ---
        # Root states
        root_quat = self.robot.data.root_quat_w
        root_lin_vel = self.robot.data.root_lin_vel_w
        root_ang_vel = self.robot.data.root_ang_vel_b
        _, current_pitch, _ = math_utils.euler_xyz_from_quat(root_quat)

        # --- 3. REWARD: Takeoff (Jump Up, Stay Upright) ---
        # We reward Z velocity, but PENALIZE tilting back too early.
        # If it tilts back on the ground, it slips.
        vertical_vel = torch.clamp(root_lin_vel[:, 2], min=0.0)
        takeoff_reward = is_takeoff * vertical_vel * 1.5  # Boost takeoff
        
        # Penalty for bad posture during takeoff (keep pitch close to 0)
        takeoff_posture_error = torch.abs(self._wrap_to_pi(current_pitch))
        takeoff_posture_reward = is_takeoff * torch.exp(-(takeoff_posture_error**2) / 0.5)

        # --- 4. REWARD: Airborne (SPIN FAST!) ---
        # We don't care about the angle here. We just want massive negative pitch velocity.
        # Target: -8.0 rad/s (approx 1.3 full rotations per second)
        target_spin = -8.0 
        current_spin = root_ang_vel[:, 1] # Pitch velocity in body frame
        
        # We use a wider sigma (4.0) so it gets points even for slow spins initially
        spin_error = torch.abs(current_spin - target_spin)
        rate_reward = is_airborne * torch.exp(-(spin_error**2) / (4.0**2))
        
        # Optional: "Tuck" Reward (Bend Knees to spin faster)
        # Penalize straight legs in the air if you want, but rate_reward usually fixes this.

        # --- 5. REWARD: Landing (Stop Spinning, Upright) ---
        # Target pitch is effectively 0 (or -2pi).
        landing_pitch_error = torch.abs(self._wrap_to_pi(current_pitch))
        landing_reward = is_landing * torch.exp(-(landing_pitch_error**2) / 0.5)
        
        # Penalize horizontal velocity on landing (stick the landing)
        landing_vel_penalty = is_landing * torch.norm(root_lin_vel[:, :2], dim=-1)
        landing_reward *= torch.exp(-landing_vel_penalty / 1.0)

        # --- 6. Penalties ---
        action_delta = self._actions - self._previous_actions
        smoothness_penalty = torch.mean(torch.square(action_delta), dim=1)

        # --- Compose Rewards ---
        rewards = {
            "takeoff_impulse": takeoff_reward * self.cfg.takeoff_vel_reward_scale,
            "takeoff_posture": takeoff_posture_reward * 0.5, # New term
            "backflip_spin": rate_reward * 3.0, # HIGH WEIGHT for spinning
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
