# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MIT Cheetah environment with HIM (History-Informed Model) support."""

from __future__ import annotations

import torch
from collections import deque

from .mit_env import MitEnv
from .mit_him_env_cfg import MitFlatHIMEnvCfg, MitRoughHIMEnvCfg


class MitHIMEnv(MitEnv):
    """MIT Cheetah environment with HIM support for history-based observations."""

    cfg: MitFlatHIMEnvCfg | MitRoughHIMEnvCfg

    def __init__(self, cfg: MitFlatHIMEnvCfg | MitRoughHIMEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # HIM-specific: Initialize observation history
        self._history_length = cfg.history_length
        # Determine base observation size (without history)
        if isinstance(cfg, MitRoughHIMEnvCfg):
            self._base_obs_size = 235  # rough terrain observation size
        else:
            self._base_obs_size = 48 + 64   # flat terrain observation size (48 + 64 lidar)

        # Initialize observation history buffer
        # Each element in history is [num_envs, base_obs_size]
        self._obs_history = deque(maxlen=self._history_length)
        
        # Initialize with zeros
        for _ in range(self._history_length):
            self._obs_history.append(torch.zeros(self.num_envs, self._base_obs_size, device=self.device))

    def _get_observations(self) -> dict:
        """Get observations with history for HIM algorithm."""
        # Get current base observation (same as parent)
        # This includes all the MIT-specific sensors: IMU, lidar, etc.
        
        # Velocity Commands [3] + Height Command [1] = [4]
        velocity_commands = self._commands[:, :3]  # [3]
        height_command = self._commands[:, 3].unsqueeze(1)  # [1] - target height

        # IMU Angular Velocity (with noise ±0.1)
        imu_ang_vel = self._imu_sensor.data.ang_vel_b  # [3]
        ang_vel_noise = torch.empty_like(imu_ang_vel).uniform_(-0.01, 0.01)
        imu_ang_vel = imu_ang_vel + ang_vel_noise  

        # IMU Orientation (with noise ±0.05)
        imu_lin_acc = self._imu_sensor.data.lin_acc_b
        orientation_noise = torch.empty_like(imu_lin_acc).uniform_(-0.03, 0.03)
        imu_lin_acc = imu_lin_acc + orientation_noise

        # Projected Gravity (with noise ±0.05)
        proj_gravity = self._robot.data.projected_gravity_b  # [3]
        gravity_noise = torch.empty_like(proj_gravity).uniform_(-0.01, 0.01)
        proj_gravity = proj_gravity + gravity_noise

        # Joint Positions (with noise ±0.01) - 使用随机化的默认位置作为参考
        joint_positions = self._robot.data.joint_pos - self._randomized_default_joint_pos  # [12]
        joint_pos_noise = torch.empty_like(joint_positions).uniform_(-0.01, 0.01)
        joint_positions = joint_positions + joint_pos_noise

        # Joint Velocities (with noise ±1.5)
        joint_velocities = self._robot.data.joint_vel  # [12]
        joint_vel_noise = torch.empty_like(joint_velocities).uniform_(-0.5, 0.5)
        joint_velocities = joint_velocities + joint_vel_noise

        # Last Actions (no noise)
        actions = self._actions  # [12]

        height_data = None
        if isinstance(self.cfg, MitRoughHIMEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)

        # 处理激光雷达深度数据
        lidar_depth_data = None
        lidar_hits = self._front_lidar.data.ray_hits_w  # [N, 64, 3] 击中点的世界坐标
        lidar_pos = self._front_lidar.data.pos_w  # [N, 3] 雷达的世界坐标
        lidar_quat = self._front_lidar.data.quat_w # [N, 4] 雷达的四元数姿态
        
        # 计算击中点相对于雷达的向量
        relative_vectors = lidar_hits - lidar_pos.unsqueeze(1)  # [N, 64, 3]
        
        # 从四元数计算雷达的X轴朝向向量
        # 四元数格式: (w, x, y, z)
        qw, qx, qy, qz = lidar_quat[:, 0], lidar_quat[:, 1], lidar_quat[:, 2], lidar_quat[:, 3]
        
        # 计算雷达X轴朝向的单位向量（X轴旋转后的方向）
        # 使用四元数旋转公式计算X轴方向向量
        forward_x = 1 - 2 * (qy * qy + qz * qz)
        forward_y = 2 * (qx * qy + qw * qz)
        forward_z = 2 * (qx * qz - qw * qy)
        
        # 组合成朝向向量 [N, 3]
        forward_vector = torch.stack([forward_x, forward_y, forward_z], dim=-1)  # [N, 3]
        
        # 计算每个击中点在雷达X轴朝向方向上的投影长度（深度）
        # 使用点积计算投影长度
        forward_vector_expanded = forward_vector.unsqueeze(1)  # [N, 1, 3]
        depth_projections = torch.sum(relative_vectors * forward_vector_expanded, dim=-1)  # [N, 64]
        
        # 处理无限远的点和负值（在雷达后方的点）
        max_range = 4
        depth_projections = torch.where(
            torch.isinf(lidar_hits).any(dim=-1),  # 检查是否有inf值
            torch.full_like(depth_projections, max_range),
            torch.clamp(depth_projections, 0.0, max_range)  # 确保深度为正值
        )
        
        # 展平深度数据
        lidar_depth_data = depth_projections.flatten(start_dim=1)  # [N, 64]
        
        # 添加传感器噪声
        depth_noise = torch.empty_like(lidar_depth_data).uniform_(-0.015, 0.015)
        lidar_depth_data = torch.clamp(lidar_depth_data + depth_noise, 0.0, max_range)
        
        #现在的顺序，从右到左，从下到上，更换成从左到右，从上到下 
        lidar_depth_data = torch.flip(lidar_depth_data, dims=[-1])  # [N, 64]

        #加大噪声，随机0到4个点 随机在0到4之间
        noise_indices = torch.randint(0, 64, (self.num_envs, 4), device=self.device)  # [N, 4] 随机选择4个索引
        noise_values = torch.empty((self.num_envs, 4), device=self.device).uniform_(0, 4)  # [N, 4] 生成对应数量的噪声值
        lidar_depth_data[torch.arange(self.num_envs, device=self.device).unsqueeze(1), noise_indices] = noise_values
        
        current_obs = torch.cat(
            [
                tensor
                for tensor in (
                    velocity_commands,    # [3]
                    height_command,       # [1] - target height
                    imu_ang_vel,         # [3]
                    imu_lin_acc,         # [3]
                    proj_gravity,        # [3]
                    joint_positions,     # [12]
                    joint_velocities,    # [12]
                    actions,            # [12]
                    height_data,         # height scanner data (for rough terrain)
                    lidar_depth_data,   # [64] - 深度投影信息
                )
                if tensor is not None
            ],
            dim=-1,
        )

        # Add current observation to history
        self._obs_history.append(current_obs.clone())

        # Create history tensor: [num_envs, history_length * base_obs_size]
        history_tensor = torch.cat(list(self._obs_history), dim=-1)

        # HIM observations include:
        # 1. policy: current observation for immediate policy decisions
        # 2. critic: full history for value function
        # 3. next_critic: next step's full history (filled during rollout)
        observations = {
            "policy": current_obs,
            "critic": history_tensor,
            "next_critic": history_tensor  # This will be updated in the next step
        }
        return observations

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environment and clear history for reset environments."""
        super()._reset_idx(env_ids)
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Clear history for reset environments by setting to zeros
        for i in range(len(self._obs_history)):
            self._obs_history[i][env_ids] = 0.0

    def get_next_critic_observations(self) -> torch.Tensor:
        """Get the next critic observations for HIM algorithm."""
        # This is called after the environment step to get next observations
        # Return the current full history as next_critic observations
        return torch.cat(list(self._obs_history), dim=-1)
