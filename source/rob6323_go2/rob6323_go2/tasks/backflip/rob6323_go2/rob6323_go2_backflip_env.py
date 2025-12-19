"""
Backflips with Cumulative Pitch Gating + Symmetry Locking + High Power
"""

from __future__ import annotations
import math
from collections.abc import Sequence
import torch
import isaaclab.utils.math as math_utils
from rob6323_go2.tasks.direct.rob6323_go2.rob6323_go2_env import Rob6323Go2Env
from .rob6323_go2_backflip_env_cfg import Rob6323Go2BackflipEnvCfg

class Rob6323Go2BackflipEnv(Rob6323Go2Env):
    cfg: Rob6323Go2BackflipEnvCfg

    def __init__(self, cfg: Rob6323Go2BackflipEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.cumulative_pitch = torch.zeros(self.num_envs, device=self.device)
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        self._episode_sums.update({
            "takeoff_power": torch.zeros(self.num_envs, device=self.device),
            "air_spin": torch.zeros(self.num_envs, device=self.device),
            "air_spin_miss": torch.zeros(self.num_envs, device=self.device),
            "land_orient": torch.zeros(self.num_envs, device=self.device),
            "land_still": torch.zeros(self.num_envs, device=self.device),
            "rotation_gate": torch.zeros(self.num_envs, device=self.device),
            "joint_limits": torch.zeros(self.num_envs, device=self.device),
            "non_pitch_penalty": torch.zeros(self.num_envs, device=self.device),
            "hip_penalty": torch.zeros(self.num_envs, device=self.device),
            "smoothness": torch.zeros(self.num_envs, device=self.device),
        })

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        phase = self._compute_phase()
        phase_features = torch.stack([torch.sin(2 * math.pi * phase), torch.cos(2 * math.pi * phase)], dim=1)
        obs["policy"] = torch.cat([obs["policy"], phase_features], dim=-1)
        return obs

    def _get_rewards(self) -> torch.Tensor:
        self.progress_buf += 1 
        
        phase = self._compute_phase()
        current_step = self.common_step_counter
        
        # Update Pitch
        self.cumulative_pitch += self.robot.data.root_ang_vel_b[:, 1] * self.step_dt
        
        # Curriculum
        curriculum_factor = torch.clamp(torch.tensor(current_step / self.cfg.curriculum_duration_steps), 0.0, 1.0).to(self.device)
        target_spin_speed = -6.0 + (curriculum_factor * -6.0) 

        # Phases
        contact_forces = torch.norm(self._contact_sensor.data.net_forces_w_history[:, -1], dim=-1).sum(dim=1)
        is_contact = contact_forces > 1.0
        
        is_airborne = (~is_contact) & (phase > 0.15) & (phase < 0.85)
        is_landing = (phase > 0.8) & is_contact
        is_takeoff = phase < 0.2

        root_lin_vel = self.robot.data.root_lin_vel_w
        root_ang_vel = self.robot.data.root_ang_vel_b

        # Takeoff
        vel_z = torch.clamp(root_lin_vel[:, 2], min=0.0)
        takeoff_reward = is_takeoff * vel_z * 2.0
        
        # Air Spin (Pitch Axis Only)
        current_spin = root_ang_vel[:, 1]
        
        # Symmetry Penalty 1
        non_pitch_rotation = torch.sum(torch.square(root_ang_vel[:, [0, 2]]), dim=1)
        non_pitch_penalty = is_airborne * non_pitch_rotation * -2.0

        # Symmetry Penalty 2
        # Forces hips to stay straight (0.0) during takeoff and air.
        abduction_joints = self.robot.data.joint_pos[:, [0, 3, 6, 9]]
        hip_penalty = (~is_landing) * torch.sum(torch.square(abduction_joints), dim=1) * -1.0

        spin_deficit = torch.clamp(current_spin - (-4.0), min=0.0)
        spin_miss_penalty = is_airborne * spin_deficit * -1.5
        
        spin_error = torch.clamp(current_spin, max=0.0) - target_spin_speed
        rate_reward = is_airborne * torch.exp(-(spin_error**2) / 5.0)

        # The Gate
        rotation_error = torch.abs(self.cumulative_pitch - (-2 * math.pi))
        rotation_gate = torch.exp(-(rotation_error**2) / 2.0)

        # Landing
        pitch_error = torch.abs(self._wrap_to_pi(math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)[1]))
        land_orient_reward = is_landing * rotation_gate * torch.exp(-(pitch_error**2) / 0.5)
        
        vel_xy = torch.norm(root_lin_vel[:, :2], dim=-1)
        land_still_reward = is_landing * rotation_gate * torch.exp(-vel_xy / 1.0)
        
        # Joint Limits
        out_of_limits = (self.robot.data.joint_pos < self.robot.data.soft_joint_pos_limits[..., 0]) | \
                        (self.robot.data.joint_pos > self.robot.data.soft_joint_pos_limits[..., 1])
        joint_limit_penalty = torch.sum(out_of_limits, dim=1).float() * -0.5

        action_smoothness = -torch.mean(torch.square(self._actions - self._previous_actions), dim=1)

        rewards = {
            "takeoff_power": takeoff_reward * 2.0, # [UPDATED] Increased per your request
            "air_spin": rate_reward * 4.0,
            "air_spin_miss": spin_miss_penalty,
            
            "non_pitch_penalty": non_pitch_penalty, 
            "hip_penalty": hip_penalty,
            
            "land_orient": land_orient_reward * 2.0,
            "land_still": land_still_reward * 1.0,
            "joint_limits": joint_limit_penalty,
            "smoothness": action_smoothness * 0.05,
        }

        for k, v in rewards.items():
            if k in self._episode_sums: self._episode_sums[k] += v
        if "rotation_gate" in self._episode_sums: self._episode_sums["rotation_gate"] += is_landing * rotation_gate

        return torch.sum(torch.stack(list(rewards.values())), dim=0)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None: env_ids = self.robot._ALL_INDICES
        
        self.cumulative_pitch[env_ids] = 0.0
        self.progress_buf[env_ids] = 0
        
        if "episode" not in self.extras: self.extras["episode"] = {}
        for k in self._episode_sums.keys():
            self.extras["episode"][k] = self._episode_sums[k][env_ids].mean()
            self._episode_sums[k][env_ids] = 0.0
        self._commands[env_ids] = 0.0

    def _compute_phase(self) -> torch.Tensor:
        return ((self.progress_buf * self.step_dt) / self.cfg.flip_period_s) % 1.0

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))