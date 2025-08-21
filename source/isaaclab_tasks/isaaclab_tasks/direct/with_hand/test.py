# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster, Imu
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG,VisualizationMarkersCfg
from .amdog_c_env_cfg import AmdogFlatEnvCfg, AmdogRoughEnvCfg


class AmdogEnv(DirectRLEnv):
    cfg: AmdogFlatEnvCfg | AmdogRoughEnvCfg

    def __init__(self, cfg: AmdogFlatEnvCfg | AmdogRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # Goal positions instead of velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)  # We'll use this for storing goal positions
        
        # Initialize goal positions (random positions within a certain radius)
       
        self._goal_positions = torch.zeros(self.num_envs, 2, device=self.device)  # x, y positions
       

        self._goal_visualizer = VisualizationMarkers(self.cfg.marker_cfg)
        self._goal_visualizer.set_visibility(True)

        self._feet_force_visualizer = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/feet_forces",
            markers={
            "fr_rl_force": sim_utils.CylinderCfg(  # 前右和后左脚 (对角组1)
                radius=0.03,
                height=0.2,
            ),}
        ))
        self._feet_force_visualizer.set_visibility(True)
        
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "heading_alignment",
                "goal_reached_count",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "standing_posture",
                "joint_position_deviation",
                "knee_joint_deviation"
                "joint_position_deviation",
                "knee_joint_deviation",
              #  "goal_directed_velocity",
                "diagonal_force_diff",
                "path_follow",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("trunk")
        #thigh_id
        self._thigh_id, _ = self._contact_sensor.find_bodies(".*thigh")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies([".*thigh", ".*calf", "trunk"])

        # 路径生成相关变量
        self._path_time = torch.zeros(self.num_envs, device=self.device)
        self._start_positions = torch.zeros(self.num_envs, 2, device=self.device)
        self._goal_positions = torch.zeros(self.num_envs, 2, device=self.device)  # x, y positions
        


        # 环境级别的路径配置
        self._env_path_types = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # 路径参数 (可以为每个环境单独配置)
        self._env_path_speeds = torch.ones(self.num_envs, device=self.device) * self.cfg.path_speed
        self._env_path_amplitudes = torch.ones(self.num_envs, device=self.device) * self.cfg.path_amplitude
        self._env_path_frequencies = torch.ones(self.num_envs, device=self.device) * self.cfg.path_frequency
        
        # 直线路径方向 (对于直线路径)
        self._env_path_directions = torch.zeros(self.num_envs, 2, device=self.device)
        self._env_path_directions[:, 0] = 1.0  # 默认x方向
        
        # 随机分配路径类型
        self._assign_random_path_types()

        # 初始化起始位置（后续路径相对于此位置生成）
        self._goal_visualizer = VisualizationMarkers(self.cfg.marker_cfg)
        self._goal_visualizer.set_visibility(True)
        
        # 样本初始目标位置
        self._initialize_path()


    def _update_feet_force_visualization(self):
        """更新足部接触力的可视化"""
        # 获取足部接触力数据
        foot_contact_forces = self._contact_sensor.data.net_forces_w_history[:, -1, self._feet_ids]  # [num_envs, 4, 3]
        foot_forces_magnitude = torch.norm(foot_contact_forces, dim=2)  # [num_envs, 4]
        
        body_feet_ids = [self._robot.data.body_names.index("FL_foot"), self._robot.data.body_names.index("FR_foot"),
                         self._robot.data.body_names.index("RL_foot"), self._robot.data.body_names.index("RR_foot")]
        # 获取足部位置
        foot_positions = self._robot.data.body_pos_w[:, body_feet_ids]  # [num_envs, 4, 2]
        
        # 计算可视化参数
        max_force = 300.0  # 力的最大值，用于归一化
        min_height = 0.05  # 最小高度
        max_height = 0.5   # 最大高度
        
        # 创建可视化对象
        num_envs = self.num_envs
        markers_pos = []   # 所有标记的位置
        markers_scale = [] # 所有标记的缩放
        markers_idx = []   # 所有标记的类型索引
        
        # 前左和后右脚 (对角组1) - 绿色
        for i in range(num_envs):
            # 前左脚 (索引0)
            fl_force = foot_forces_magnitude[i, 0]
            if fl_force > 5.0:  # 只在力大于阈值时显示
                fl_pos = foot_positions[i, 0]
                # 根据力的大小计算高度
                height = min_height + (max_height - min_height) * torch.min(fl_force / max_force, torch.tensor(1.0, device=self.device))
                # 添加到标记列表
                markers_pos.append(fl_pos + torch.tensor([0, 0, height/2], device=self.device))
                markers_scale.append(torch.tensor([1.0, 1.0, height/0.2], device=self.device))
                markers_idx.append(0)  # 使用绿色圆柱体
            
            # 后右脚 (索引3)
            rr_force = foot_forces_magnitude[i, 3]
            if rr_force > 5.0:
                rr_pos = foot_positions[i, 3]
                height = min_height + (max_height - min_height) * torch.min(rr_force / max_force, torch.tensor(1.0, device=self.device))
                markers_pos.append(rr_pos + torch.tensor([0, 0, height/2], device=self.device))
                markers_scale.append(torch.tensor([1.0, 1.0, height/0.2], device=self.device))
                markers_idx.append(0)  # 使用绿色圆柱体
        
        # 如果有标记需要可视化
        if markers_pos:
            # 将列表转换为张量
            positions_tensor = torch.stack(markers_pos)
            scales_tensor = torch.stack(markers_scale)
            indices_tensor = torch.tensor(markers_idx, device=self.device)
            
            # 可视化
            self._feet_force_visualizer.visualize(
                translations=positions_tensor,
                scales=scales_tensor,
                marker_indices=indices_tensor
            )
        else:
            # 如果没有足部力，清空可视化
            empty_tensor = torch.zeros((1, 3), device=self.device)
            self._feet_force_visualizer.visualize(
                translations=empty_tensor,
                scales=empty_tensor,
                marker_indices=torch.zeros(1, dtype=torch.long, device=self.device)
            )


    def _assign_random_path_types(self, env_ids=None):
        """为每个环境随机分配路径类型"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # 基于概率分配路径类型
        path_type_indices = torch.multinomial(
            torch.tensor(self.cfg.path_types_probs, device=self.device),
            len(env_ids),
            replacement=True
        )
        self._env_path_types[env_ids] = path_type_indices
        
        # 为直线路径随机生成方向
        line_envs = env_ids[path_type_indices == 0]  # 假设"line"的索引是0
        if len(line_envs) > 0:
            # 生成随机角度
            random_angles = torch.rand(len(line_envs), device=self.device) * 2 * torch.pi
            # 转换为单位向量
            self._env_path_directions[line_envs, 0] = torch.cos(random_angles)
            self._env_path_directions[line_envs, 1] = torch.sin(random_angles)
        
        # 可选：为每个环境随机化路径参数
        # 比如随机化路径速度
        self._env_path_speeds[env_ids] = self.cfg.path_speed * (0.8 + 0.4 * torch.rand(len(env_ids), device=self.device))
        
        # 随机化路径振幅
        self._env_path_amplitudes[env_ids] = self.cfg.path_amplitude * (0.7 + 0.6 * torch.rand(len(env_ids), device=self.device))
        
        # 随机化正弦波频率
        sine_envs = env_ids[path_type_indices == 1]  # 假设"sine"的索引是1
        if len(sine_envs) > 0:
            self._env_path_frequencies[sine_envs] = self.cfg.path_frequency * (0.5 + 1.0 * torch.rand(len(sine_envs), device=self.device))


    def _initialize_path(self):
        """初始化路径，设置起始位置"""
        # 获取所有机器人的初始位置
        self._start_positions = self._robot.data.root_pos_w[:, :2].clone()
        # 重置路径时间
        self._path_time.zero_()
        # 生成初始目标点位置
        self._update_path_goals()
        # 更新可视化
        self._update_goal_visualization()

    def _update_path_goals(self):
        """基于当前时间更新目标位置，支持多种路径类型"""
        # 为每种路径类型创建掩码
        line_mask = self._env_path_types == 0
        sine_mask = self._env_path_types == 1
        circle_mask = self._env_path_types == 2
        square_mask = self._env_path_types == 3
        
        # 处理直线路径环境
        if torch.any(line_mask):
            line_envs = torch.nonzero(line_mask).squeeze(-1)
            distance = self._path_time[line_envs] * self._env_path_speeds[line_envs]
            offset = distance.unsqueeze(1) * self._env_path_directions[line_envs]
            self._goal_positions[line_envs] = self._start_positions[line_envs] + offset
        
        # 处理正弦路径环境
        if torch.any(sine_mask):
            sine_envs = torch.nonzero(sine_mask).squeeze(-1)
            t = self._path_time[sine_envs]
            
            # 主方向前进
            x = self._start_positions[sine_envs, 0] + t * self._env_path_speeds[sine_envs]
            
            # Y方向做正弦运动
            y = self._start_positions[sine_envs, 1] + self._env_path_amplitudes[sine_envs] * torch.sin(
                2 * torch.pi * self._env_path_frequencies[sine_envs] * t
            )
            
            self._goal_positions[sine_envs] = torch.stack([x, y], dim=1)
        
        # 处理圆形路径环境
        if torch.any(circle_mask):
            circle_envs = torch.nonzero(circle_mask).squeeze(-1)
            t = self._path_time[circle_envs]
            angle = t * self._env_path_speeds[circle_envs]
            
            x = self._start_positions[circle_envs, 0] + self._env_path_amplitudes[circle_envs] * torch.cos(angle)
            y = self._start_positions[circle_envs, 1] + self._env_path_amplitudes[circle_envs] * torch.sin(angle)
            
            self._goal_positions[circle_envs] = torch.stack([x, y], dim=1)
        
        # 处理方形路径环境
        if torch.any(square_mask):
            square_envs = torch.nonzero(square_mask).squeeze(-1)
            t = self._path_time[square_envs]
            
            # 为每个环境计算周期和相关参数
            side_lengths = self._env_path_amplitudes[square_envs] * 2
            periods = 4 * side_lengths / self._env_path_speeds[square_envs]
            t_cycles = t % periods
            side_times = periods / 4
            
            # 确定每个环境当前在哪条边上
            side_indices = (t_cycles / side_times).long()
            progress = (t_cycles % side_times) / side_times
            
            # 准备结果数组
            x = self._start_positions[square_envs, 0].clone()
            y = self._start_positions[square_envs, 1].clone()
            
            # 处理每条边上的环境
            for side_idx in range(4):
                side_mask = side_indices == side_idx
                if torch.any(side_mask):
                    edge_envs = torch.nonzero(side_mask).squeeze(-1)
                    amplitudes = self._env_path_amplitudes[square_envs][edge_envs]
                    progs = progress[edge_envs]
                    
                    if side_idx == 0:  # 上边
                        x[edge_envs] += -amplitudes + 2 * amplitudes * progs
                        y[edge_envs] += amplitudes
                    elif side_idx == 1:  # 右边
                        x[edge_envs] += amplitudes
                        y[edge_envs] += amplitudes - 2 * amplitudes * progs
                    elif side_idx == 2:  # 下边
                        x[edge_envs] += amplitudes - 2 * amplitudes * progs
                        y[edge_envs] += -amplitudes
                    else:  # 左边
                        x[edge_envs] += -amplitudes
                        y[edge_envs] += -amplitudes + 2 * amplitudes * progs
            
            self._goal_positions[square_envs] = torch.stack([x, y], dim=1)



    def _update_goal_visualization(self):
        """Update the visualization of goal positions with scale pulsing effect for close goals."""
        # Convert 2D positions to 3D by setting a fixed height
        goal_positions_3d = torch.zeros((self.num_envs, 3), device=self.device)
        goal_positions_3d[:, :2] = self._goal_positions
        
        # Set height slightly above ground
        terrain_heights = torch.zeros_like(self._goal_positions[:, 0])
        if hasattr(self, '_terrain') and hasattr(self._terrain, 'get_heights'):
            terrain_heights = self._terrain.get_heights(self._goal_positions)
        
        goal_positions_3d[:, 2] = terrain_heights + self.cfg.goal_height  # self.cfg.goal_height meters above ground
        
        # Calculate distances to goals to determine scales
        robot_pos_xy = self._robot.data.root_pos_w[:, :2]
        goal_directions = self._goal_positions - robot_pos_xy
        goal_distances = torch.norm(goal_directions, dim=1)
        
        # Default scale for all goals
        base_scale = 1.0
        scales = torch.ones((self.num_envs, 3), device=self.device) * base_scale
        
        # Find goals that are close to being reached
        close_goals = goal_distances < self.cfg.goal_close_threshold
        
        # Create pulsing effect for close goals (if any)
        if torch.any(close_goals):
            # Calculate pulsing factor based on time/steps
            # This creates a value that oscillates between 0.5 and 1.5
            pulse_factor = 0.5 * (1.0 + torch.sin(self.episode_length_buf.float() * 0.2)) + 0.5
            
            # Apply the pulse factor to all close goals
            for i in torch.nonzero(close_goals).squeeze(-1):
                scales[i] = scales[i] * pulse_factor[i]
        
        # Update the visualization markers with positions and scales
        self._goal_visualizer.visualize(translations=goal_positions_3d, scales=scales)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # Add IMU sensor
        self._imu_sensor = Imu(self.cfg.imu)
        self.scene.sensors["imu_sensor"] = self._imu_sensor

        if isinstance(self.cfg, AmdogRoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
        
        # 更新路径时间和目标位置
        self._path_time += self.step_dt
        self._update_path_goals()
        self._update_goal_visualization()
        # 添加足部力可视化更新（降低更新频率以提高性能）
        if self.episode_length_buf[0] % 20 == 0:
            self._update_feet_force_visualization()


    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        
        # Add noise to observations
        device = self._robot.device
        
        # Calculate relative goal direction in local frame
        # Fix: Use correct properties from robot.data
        robot_pos_xy = self._robot.data.root_pos_w[:, :2]  # Position in world frame
        robot_rot = self._robot.data.root_quat_w  # Quaternion in world frame
        
        # Calculate heading (yaw) of the robot
        # Extract yaw from quaternion
        qw, qx, qy, qz = robot_rot[:, 0], robot_rot[:, 1], robot_rot[:, 2], robot_rot[:, 3]
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        
        # Calculate goal direction in world frame
        goal_direction_world = self._goal_positions - robot_pos_xy
        
        # Transform goal direction to robot's local frame
        cos_yaw = torch.cos(-yaw).unsqueeze(1)
        sin_yaw = torch.sin(-yaw).unsqueeze(1)
        
        goal_dir_local_x = cos_yaw * goal_direction_world[:, 0:1] - sin_yaw * goal_direction_world[:, 1:2]
        goal_dir_local_y = sin_yaw * goal_direction_world[:, 0:1] + cos_yaw * goal_direction_world[:, 1:2]
        
        # Calculate heading angle error (angle between robot's forward direction and goal)
        heading_angle_error = torch.atan2(goal_dir_local_y, goal_dir_local_x)
        
        # IMU Angular Velocity
        imu_ang_vel = self._imu_sensor.data.ang_vel_b  # [3]
        # ang_vel_noise = torch.empty_like(imu_ang_vel).uniform_(-0.1, 0.1)
        # imu_ang_vel = imu_ang_vel + ang_vel_noise  

        # IMU Linear Acceleration
        imu_lin_acc = self._imu_sensor.data.lin_acc_b

        # Projected Gravity
        proj_gravity = self._robot.data.projected_gravity_b  # [3]
        # gravity_noise = torch.empty_like(proj_gravity).uniform_(-0.1, 0.1)
        # proj_gravity = proj_gravity + gravity_noise

        # Joint Positions
        joint_positions = self._robot.data.joint_pos - self._robot.data.default_joint_pos  # [12]
        # joint_pos_noise = torch.empty_like(joint_positions).uniform_(-0.1, 0.1)
        # joint_positions = joint_positions + joint_pos_noise

        # Joint Velocities
        joint_velocities = self._robot.data.joint_vel  # [12]
        # joint_vel_noise = torch.empty_like(joint_velocities).uniform_(-1.5, 1.5)
        # joint_velocities = joint_velocities + joint_vel_noise

        # Last Actions (no noise)
        actions = self._actions  # [12]
            
        obs = torch.cat(
            [
                heading_angle_error,   # [1] just the angle error to goal
                imu_ang_vel,          # [3]
                imu_lin_acc,          # [3]
                proj_gravity,         # [3]
                joint_positions,      # [12]
                joint_velocities,     # [12]
                actions,              # [12]
            ],
            dim=-1,
        )

        observations = {"policy": obs}  # Total dimensions: 46
        return observations

    def _get_rewards(self) -> torch.Tensor:
        robot_pos_xy = self._robot.data.root_pos_w[:, :2]
        goal_direction = self._goal_positions - robot_pos_xy
        goal_distance = torch.norm(goal_direction, dim=1)
        
        # Robot's forward direction in world frame
        # Extract forward direction from quaternion - fix: use root_quat_w
        robot_rot = self._robot.data.root_quat_w
        qw, qx, qy, qz = robot_rot[:, 0], robot_rot[:, 1], robot_rot[:, 2], robot_rot[:, 3]
        
        # Calculate forward vector from quaternion
        forward_x = 1 - 2 * (qy * qy + qz * qz)
        forward_y = 2 * (qx * qy + qw * qz)
        forward = torch.stack([forward_x, forward_y], dim=1)
        
        # Normalize vectors for dot product
        forward_norm = torch.norm(forward, dim=1, keepdim=True) + 1e-6
        forward_normalized = forward / forward_norm
        
        goal_dir_norm = torch.norm(goal_direction, dim=1, keepdim=True) + 1e-6
        goal_dir_normalized = goal_direction / goal_dir_norm
        
        # Heading alignment via dot product (-1 to 1, where 1 is perfect alignment)
        heading_alignment = torch.sum(forward_normalized * goal_dir_normalized, dim=1)
        
            # 目标达到检测，但不刷新目标
        goal_reached = goal_distance < self.cfg.goal_threshold
        
        # 目标完成奖励 - 仍然给予奖励，但不刷新目标
        goal_completion_reward = torch.zeros_like(goal_distance)
        goal_completion_reward[goal_reached] = 5.0

        
        # Heading alignment reward
        heading_reward_scale = 0.5
        heading_reward = (heading_alignment + 1.0) * 0.5 * heading_reward_scale
        
        # Goal completion reward
        goal_completion_reward = torch.zeros_like(goal_distance)
        goal_completion_reward[goal_reached] = 5.0
        
        # NEW: Goal-directed velocity reward
        # Get robot velocity in world frame
        robot_vel_xy = self._robot.data.root_lin_vel_w[:, :2]
        
        # Project velocity onto goal direction to get component of velocity towards goal
        goal_dir_normalized_safe = torch.where(
            goal_dir_norm > 1e-6, 
            goal_direction / goal_dir_norm, 
            torch.zeros_like(goal_direction)
        )
        
        # Calculate goal-directed velocity (dot product of velocity and goal direction)
      # Replace lines 288-293 with this implementation:

        # Calculate goal-directed velocity (dot product of velocity and goal direction)
        goal_directed_vel = torch.sum(robot_vel_xy * goal_dir_normalized_safe, dim=1)

        # Get velocity magnitude
        vel_magnitude = torch.norm(robot_vel_xy, dim=1)

        # Calculate the cosine of the angle between velocity and goal direction
        # (only when the robot is actually moving)
        cos_angle = torch.zeros_like(goal_directed_vel)
        moving_mask = vel_magnitude > 0.1  # Only consider robots that are moving
        if torch.any(moving_mask):
            cos_angle[moving_mask] = goal_directed_vel[moving_mask] / vel_magnitude[moving_mask]

        # Scale velocity reward - positive for moving towards goal, negative for moving away
        vel_reward_scale = 2.0  # Adjust this value based on desired importance
        #direction_factor = (cos_angle + 1.0) / 2.0  # Map from [-1,1] to [0,1]
        speed_factor = torch.clamp(vel_magnitude, 0.0, 3.0) / 3.0  # Normalized speed between [0,1]

        # Combined reward that considers both speed and direction
        velocity_reward = vel_reward_scale * speed_factor * cos_angle * 3.0
        
        # Calculate goal-directed velocity (dot product of velocity and goal direction)
        goal_directed_vel = torch.sum(robot_vel_xy * goal_dir_normalized_safe, dim=1)
        
        # Get velocity magnitude
        vel_magnitude = torch.norm(robot_vel_xy, dim=1)
        
        # Calculate the cosine of the angle between velocity and goal direction
        # (only when the robot is actually moving)
        cos_angle = torch.zeros_like(goal_directed_vel)
        moving_mask = vel_magnitude > 0.1  # Only consider robots that are moving
        if torch.any(moving_mask):
            cos_angle[moving_mask] = goal_directed_vel[moving_mask] / vel_magnitude[moving_mask]
        
        # Scale velocity reward - positive for moving towards goal, negative for moving away
        vel_reward_scale = 2.0  # Adjust this value based on desired importance
        direction_factor = (cos_angle + 1.0) / 2.0  # Map from [-1,1] to [0,1]
        speed_factor = torch.clamp(vel_magnitude, 0.0, 3.0) / 3.0  # Normalized speed between [0,1]
        
        # Combined reward that considers both speed and direction
        velocity_reward = vel_reward_scale * speed_factor * direction_factor * 3.0
                    
        # Original penalties
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        
        # Feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        # No longer condition on command velocity, use a constant value for air time
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1)
        
        # Undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        
        # Flat orientation penalty
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        
        # Standing posture penalty - now tied to goal distance rather than velocity
        is_standing = goal_distance < 1.0  # When close to goal, encourage standing position
        joint_deviation = torch.sum(
            torch.square(self._robot.data.joint_pos - self._robot.data.default_joint_pos),
            dim=1
        )
        standing_posture_penalty = joint_deviation * is_standing.float()
        
        # Joint position deviation penalties
        joint_position_deviation = torch.sum(
            torch.square(self._robot.data.joint_pos - self._robot.data.default_joint_pos),
            dim=1
        )
        
        knee_joint_indices = [8, 9, 10, 11]
        knee_joint_deviation = torch.sum(
            torch.square(
                self._robot.data.joint_pos[:, knee_joint_indices] - 
                self._robot.data.default_joint_pos[:, knee_joint_indices]
            ),
            dim=1
        )

        # 获取足部接触力
        foot_contact_forces = self._contact_sensor.data.net_forces_w_history[:, -1, self._feet_ids]  # 最新的力数据
        foot_forces_magnitude = torch.norm(foot_contact_forces, dim=2)  # [num_envs, 4]
        
        # 假设四只脚的顺序是: 前左(FL)，前右(FR)，后左(RL)，后右(RR)
        # 计算对角脚的力差异: |FL-RR| 和 |FR-RL|
        fl_rr_diff = torch.abs(foot_forces_magnitude[:, 0] - foot_forces_magnitude[:, 3])
        fr_rl_diff = torch.abs(foot_forces_magnitude[:, 1] - foot_forces_magnitude[:, 2])
        
        # 总的对角脚力差异
        diagonal_force_diff = fl_rr_diff + fr_rl_diff
        diagonal_force_diff = diagonal_force_diff/100
        #l2
        diagonal_force_diff = torch.square(diagonal_force_diff)

    
        # 获取路径增量方向（目标点的移动方向）
        current_goals = self._goal_positions.clone()
        previous_time = self._path_time - self.step_dt
        
        # 临时保存当前时间
        current_time = self._path_time.clone()
        
        # 回退路径时间来计算先前目标
        self._path_time = previous_time
        self._update_path_goals()
        previous_goals = self._goal_positions.clone()
        
        # 恢复当前时间
        self._path_time = current_time
        self._update_path_goals()
        
        # 计算路径方向
        path_direction = current_goals - previous_goals
        path_direction_norm = torch.norm(path_direction, dim=1, keepdim=True) + 1e-6
        path_direction_normalized = path_direction / path_direction_norm
        
        # 计算机器狗沿路径方向的速度分量
        path_aligned_velocity = torch.sum(self._robot.data.root_lin_vel_w[:, :2] * path_direction_normalized, dim=1)
        
        # 路径跟随奖励
        path_follow_reward = torch.clamp(path_aligned_velocity, 0.0, 2.0)
        
        
        rewards = {
            "heading_alignment": heading_reward * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "goal_reached_count": goal_completion_reward * self.cfg.touch_goal_reward_scale*self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "standing_posture": standing_posture_penalty * self.cfg.standing_posture_penalty * self.step_dt,
            "joint_position_deviation": joint_position_deviation * self.cfg.joint_position_deviation_scale * self.step_dt,
            "knee_joint_deviation": knee_joint_deviation * self.cfg.knee_joint_deviation_scale * self.step_dt, 
           # "goal_directed_velocity": velocity_reward * self.cfg.lin_vel_reward_scale * self.step_dt,
            "diagonal_force_diff": diagonal_force_diff * self.cfg.diagonal_force_diff_reward_scale * self.step_dt,
            "path_follow": path_follow_reward * self.cfg.path_follow_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        #thigh_contact = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._thigh_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        #died = died | thigh_contact
        return died, time_out
       # return time_out, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # 重置路径时间和起始位置
        self._path_time[env_ids] = 0.0
        self._start_positions[env_ids] = self._robot.data.root_pos_w[env_ids, :2].clone()
        
        # 为重置的环境重新分配路径类型
        self._assign_random_path_types(env_ids)
        
        # 更新目标位置和可视化
        self._update_path_goals()
        self._update_goal_visualization()
     # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
