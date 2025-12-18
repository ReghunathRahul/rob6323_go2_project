"""
Backflips with Cumulative Pitch Gating
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
    Task for training periodic backflips using cumulative pitch gating.
    """

    cfg: Rob6323Go2BackflipEnvCfg

    def __init__(self, cfg: Rob6323Go2BackflipEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # [NEW] Track total rotation to prevent "safety jumps"
        self.cumulative_pitch = torch.zeros(self.num_envs, device=self.device)
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # Initialize logging
        self._episode_sums.update({
            "takeoff_power": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "takeoff_stable": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "air_spin": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "air_spin_miss": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "air_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "land_orient": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "land_still": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "smoothness": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "rotation_gate": torch.zeros(self.num_envs, dtype=torch.float, device=self.device), # [NEW] Debug metric
        })

    def _get_observations(self) -> dict:
        observations = super()._get_observations()
        phase = self._compute_phase()
        phase_features = torch.stack([torch.sin(2 * math.pi * phase), torch.cos(2 * math.pi * phase)], dim=1)
        observations["policy"] = torch.cat([observations["policy"], phase_features], dim=-1)
        return observations

    def _get_rewards(self) -> torch.Tensor:
        phase = self._compute_phase()
        current_step = self.common_step_counter
        
        # Backflip = Negative pitch velocity
        root_ang_vel = self.robot.data.root_ang_vel_b
        self.cumulative_pitch += root_ang_vel[:, 1] * self.step_dt
        
        # Curriculum
        curriculum_duration = 3_000_000.0 
        curriculum_factor = torch.clamp(torch.tensor(current_step / curriculum_duration), 0.0, 1.0).to(self.device)
        target_spin_speed = -6.0 + (curriculum_factor * -4.0) 

        # Phases
        is_takeoff = phase < 0.2
        is_airborne = (phase >= 0.2) & (phase <= 0.75)
        is_landing = phase > 0.75
        
        root_lin_vel = self.robot.data.root_lin_vel_w
        root_quat = self.robot.data.root_quat_w
        _, current_pitch, _ = math_utils.euler_xyz_from_quat(root_quat)


        # 1. Takeoff
        vel_z = torch.clamp(root_lin_vel[:, 2], min=0.0)
        takeoff_reward = is_takeoff * vel_z * 2.0
        
        # Relaxed takeoff stability 
        takeoff_stable_reward = is_takeoff * torch.exp(-(root_ang_vel[:, 1]**2) / 2.0)

        # 2. Air Spin
        current_spin = root_ang_vel[:, 1]

        # Penalty for not spinning fast enough
        min_required_spin = -4.0  
        spin_deficit = torch.clamp(min_required_spin - current_spin, min=0.0)
        spin_miss_penalty = is_airborne * spin_deficit
        
        # Reward for hitting target speed
        spin_error = current_spin - target_spin_speed
        rate_reward = is_airborne * torch.exp(-(spin_error**2) / 5.0)
        
        # Air Ground Contact Penalty
        contact_forces = torch.norm(self._contact_sensor.data.net_forces_w_history[:, -1], dim=-1).sum(dim=1)
        air_penalty = is_airborne * (contact_forces > 1.0).float() * -0.5

        # We require a full backward rotation (-2pi approx -6.28 rads)
        required_rotation = -2 * math.pi
        rotation_error = torch.abs(self.cumulative_pitch - required_rotation)
        
        # This gate is 1.0 if we flipped, 0.0 if we just jumped.
        rotation_gate = torch.exp(-(rotation_error**2) / 2.0)

        # Multiply landing rewards by the gate!
        pitch_error = torch.abs(self._wrap_to_pi(current_pitch))
        land_orient_reward = is_landing * rotation_gate * torch.exp(-(pitch_error**2) / 0.5)
        
        vel_xy = torch.norm(root_lin_vel[:, :2], dim=-1)
        land_still_reward = is_landing * rotation_gate * torch.exp(-vel_xy / 1.0)

        action_smoothness = -torch.mean(torch.square(self._actions - self._previous_actions), dim=1)

        rewards = {
            "takeoff_power": takeoff_reward * 1.0,
            "takeoff_stable": takeoff_stable_reward * 0.5,

            "air_spin": rate_reward * 4.0,
            "air_spin_miss": -spin_miss_penalty * 1.5,
            "air_penalty": air_penalty,

            "land_orient": land_orient_reward * 2.0, # Increased weight
            "land_still": land_still_reward * 1.0,

            "smoothness": action_smoothness * 0.05
        }

        # Update accumulators for logging
        for key, value in rewards.items():
            if key in self._episode_sums:
                self._episode_sums[key] += value
 
        if "rotation_gate" in self._episode_sums:
            self._episode_sums["rotation_gate"] += is_landing * rotation_gate

        return torch.sum(torch.stack(list(rewards.values())), dim=0)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        self.cumulative_pitch[env_ids] = 0.0

        if "episode" not in self.extras:
            self.extras["episode"] = {}

        for key in self._episode_sums.keys():
            self.extras["episode"][key] = self._episode_sums[key][env_ids].mean()
            self._episode_sums[key][env_ids] = 0.0

        self._commands[env_ids] = 0.0

    def _compute_phase(self) -> torch.Tensor:
        phase = (self.progress_buf * self.step_dt) / self.cfg.flip_period_s
        return phase % 1.

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))