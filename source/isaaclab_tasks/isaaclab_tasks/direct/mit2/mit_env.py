# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
from typing import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster, Imu
from .mit_env_cfg import MitFlatEnvCfg, MitRoughEnvCfg


class MitEnv(DirectRLEnv):
    cfg: MitFlatEnvCfg | MitRoughEnvCfg

    def __init__(self, cfg: MitFlatEnvCfg | MitRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "upward_reward",
                "joint_vel_l2",
                "stand_still_reward",  # 新增：停止时静止奖励
            ]
        }
        self._fallen_timeout =2.5  # 倒地超过3秒触发重置
        self._fallen_timer = torch.zeros(self.num_envs, device=self.device)  # 跟踪每个环境倒地的持续时间
        
        # 命令重新采样相关变量
        self._command_resample_time_min = 3  # 最小重新采样间隔（秒）
        self._command_resample_time_max = 10.0  # 最大重新采样间隔（秒）
        self._command_timer = torch.zeros(self.num_envs, device=self.device)  # 命令计时器
        self._command_timeout = torch.zeros(self.num_envs, device=self.device)  # 每个环境的命令超时时间
        
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*_foot")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies([".*_thigh",".*_calf"]) #, "base"

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

         # Add IMU sensor
        self._imu_sensor = Imu(self.cfg.imu)
        self.scene.sensors["imu_sensor"] = self._imu_sensor

        if isinstance(self.cfg, MitRoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner

            self._front_lidar = RayCaster(self.cfg.front_lidar)
            self.scene.sensors["front_lidar"] = self._front_lidar

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
        #把cfg.action_scale列表转为 torch.Tensor
        action_scales = torch.tensor(self.cfg.action_scale, device=self.device).view(1, -1)
        self._processed_actions = action_scales * self._actions + self._robot.data.default_joint_pos
        
        # 更新命令重新采样
        self._update_command_resample()

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        
        # Add noise to observations
        device = self._robot.device
        
        # Commands (no noise)
        velocity_commands = self._commands  # [3]

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

        # Joint Positions (with noise ±0.01)
        joint_positions = self._robot.data.joint_pos - self._robot.data.default_joint_pos  # [12]
        joint_pos_noise = torch.empty_like(joint_positions).uniform_(-0.01, 0.01)
        joint_positions = joint_positions + joint_pos_noise

        # Joint Velocities (with noise ±1.5)
        joint_velocities = self._robot.data.joint_vel  # [12]
        joint_vel_noise = torch.empty_like(joint_velocities).uniform_(-0.5, 0.5)
        joint_velocities = joint_velocities + joint_vel_noise

        # Last Actions (no noise)
        actions = self._actions  # [12]

        height_data = None
        if isinstance(self.cfg, MitRoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
        # 处理激光雷达深度数据
        lidar_depth_data = None
        if isinstance(self.cfg, MitRoughEnvCfg): 
            lidar_hits = self._front_lidar.data.ray_hits_w  # [N, 64, 3] 击中点的世界坐标
            lidar_pos = self._front_lidar.data.pos_w  # [N, 3] 雷达的世界坐标
            lidar_quat = self._front_lidar.data.quat_w  # [N, 4] 雷达的四元数姿态
            
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
            
            # 应用距离映射函数，突出0-1范围的重要性
            #lidar_depth_data = self._apply_distance_mapping(lidar_depth_data, max_range)

            #现在的顺序，从右到左，从下到上，更换成从左到右，从上到下 
            lidar_depth_data = torch.flip(lidar_depth_data, dims=[-1])  # [N, 64]

            #加大噪声，随机0到4个点 随机在0到4之间
            noise_indices = torch.randint(0, 64, (self.num_envs, 4), device=self.device)  # [N, 4] 随机选择4个索引
            noise_values = torch.empty((self.num_envs, 4), device=self.device).uniform_(0, 4)  # [N, 4] 生成对应数量的噪声值
            lidar_depth_data[torch.arange(self.num_envs, device=self.device).unsqueeze(1), noise_indices] = noise_values
            
            #lidar 全部置为0
            #lidar_depth_data = torch.zeros_like(lidar_depth_data)  # [N, 64]
            #print("Lidar Depth Data 0:", lidar_depth_data[0])  # 打印第一个环境的深度数据
        else:
            lidar_depth_data = torch.zeros(self.num_envs, 64, device=self.device)  # [N, 64] 全部置为0
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    velocity_commands,    # [3]
                    imu_ang_vel,         # [3]
                    imu_lin_acc,         # [3]
                    proj_gravity,        # [3]
                    joint_positions,     # [12]
                    joint_velocities,    # [12]
                    actions,            # [12]
                    #height_data,         # 
                    lidar_depth_data,   # [64] - 深度投影信息
                )
                if tensor is not None
            ],
            dim=-1,
        )
        teacher = torch.cat(
            [
                tensor
                for tensor in (
                    velocity_commands,    # [3]
                    imu_ang_vel,         # [3]
                    imu_lin_acc,         # [3]
                    proj_gravity,        # [3]
                    joint_positions,     # [12]
                    joint_velocities,    # [12]
                    actions,            # [12]
                    height_data,         # 
                    #lidar_depth_data,   # [64] - 深度投影信息
                )
                if tensor is not None
            ],
            dim=-1,
        )
        #observations = {"policy": obs, "teacher": teacher}
        #observations = {"policy": teacher}
        observations = {"policy": obs }
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        
        # joint velocity penalty (using L2 squared kernel like ManagerBasedRLEnv)
        joint_vel_penalty = self._compute_joint_vel_penalty()

        upward_reward = self._compute_upward_reward()
        
        # stand still reward - when command velocity is low and all feet are on ground
        stand_still_reward = self._compute_stand_still_reward()
        
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "upward_reward": upward_reward * 0 * self.step_dt,
            "joint_vel_l2": joint_vel_penalty * 0 * self.step_dt,
            "stand_still_reward": stand_still_reward * 0 * self.step_dt,  # 使用固定的奖励系数0.5
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward


    # def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    #     # 普通的超时检测
    #     time_out = self.episode_length_buf >= self.max_episode_length - 1
        
    #     # 检测机器人的姿态
    #     projected_gravity_z = self._robot.data.projected_gravity_b[:, 2]
    #     safe_values = torch.clamp(projected_gravity_z, -1.0, 1.0)
    #     angle_rad = torch.acos(-safe_values)
    #     angle_deg = angle_rad * 180.0 / torch.pi
        
    #     # 判断是否倒地
    #     fallen_threshold = 90.0  # 超过90度认为倒地
    #     is_fallen = angle_deg > fallen_threshold
        
    #     # 更新倒地计时器
    #     # 如果倒地，则累加时间；否则重置计时器
    #     self._fallen_timer = torch.where(
    #         is_fallen,
    #         self._fallen_timer + self.step_dt,  # 累加时间
    #         torch.zeros_like(self._fallen_timer)  # 重置计时器
    #     )
        
    #     # 检查是否倒地超时
    #     fallen_timeout = self._fallen_timer >= self._fallen_timeout
        
    #     # 机器人高度过低也视为终止
    #     # robot_height = self._robot.data.root_pos_w[:, 2]
    #     # height_too_low = robot_height < -4
        
    #     # 组合终止条件：倒地超时或高度过低
    #     died = fallen_timeout #| height_too_low
        
    #     return died, time_out

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        # Check body collision
        body_collision = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return body_collision, time_out


    def _compute_upward_reward(self) -> torch.Tensor:
        """计算站立奖励，使用重力投影直接映射，45°以内的角度都获得最大奖励
        
        Returns:
            torch.Tensor: 站立奖励值
        """
        # projected_gravity_z的范围为[-1, 1]
        # -1表示完全正立(重力投影与z轴反向)，1表示完全倒立
        projected_gravity_z = self._robot.data.projected_gravity_b[:, 2]
        
        # 计算倾斜角度 (弧度)
        safe_values = torch.clamp(projected_gravity_z, -1.0, 1.0)
        angle_rad = torch.acos(-safe_values)
        angle_deg = angle_rad * 180.0 / torch.pi
        
        # 定义45度的阈值（在这个范围内获得最大奖励）
        threshold_deg = 15
        
        # 将角度映射到奖励值
        # - 角度 <= 45°: 奖励为1.0（最大值）
        # - 角度 > 45°: 根据角度线性减少到0（180°时为0）
        max_angle = 180.0
        upward_reward = torch.where(
            angle_deg <= threshold_deg,
            torch.ones_like(angle_deg),  # 45°以内获得满分奖励
            1.0 - (angle_deg - threshold_deg) / (max_angle - threshold_deg)  # 45°以外线性衰减
        )
        
        # 确保奖励值在[0,1]范围内
        upward_reward = torch.clamp(upward_reward, 0.0, 1.0)
        
        # 添加非线性奖励曲线，使用平方根函数而不是平方函数
        # 平方根函数特性：增长较为平缓，随着输入值的增加增长速度减慢
        # 对于接近0的输入会放大，对于接近1的输入会减小
        upward_reward = torch.sqrt(upward_reward)  # 平方根函数替代平方函数
        
        return upward_reward
    
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
        self._fallen_timer[env_ids] = 0.0
        
        # 重置命令计时器和超时时间
        self._command_timer[env_ids] = 0.0
        # 为重置的环境设置新的重新采样间隔时间（1-10秒随机）
        self._command_timeout[env_ids] = torch.zeros_like(self._command_timeout[env_ids]).uniform_(
            self._command_resample_time_min, self._command_resample_time_max
        )
        
        # 只在初始化时采样一次命令，后续由_update_command_resample自动处理
        # X轴: -1 到 1
        self._commands[env_ids, 0] = torch.zeros_like(self._commands[env_ids, 0]).uniform_(-0.3, 0.3)
        # Y轴: -1 到 1
        self._commands[env_ids, 1] = torch.zeros_like(self._commands[env_ids, 1]).uniform_(-0.3, 0.3)
        # Z轴(角速度): -1.0 到 1.0
        self._commands[env_ids, 2] = torch.zeros_like(self._commands[env_ids, 2]).uniform_(-1, 1)
        
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
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

    def _apply_distance_mapping(self, distances: torch.Tensor, max_range: float = 4.0) -> torch.Tensor:
        """
        将雷达距离数据从0-4范围映射到更适合的范围，突出0-1之间的重要距离值
        
        提供多种映射策略：
        1. 分段线性映射：0-1范围保持细粒度，1-4范围压缩
        2. 对数映射：自然突出近距离值
        3. 指数衰减映射：平滑过渡，突出近距离
        
        Args:
            distances: 原始距离数据 [N, num_rays]
            max_range: 最大距离范围
            
        Returns:
            torch.Tensor: 映射后的距离数据，范围在[0,1]
        """
        # 策略1：分段线性映射 (推荐)
        # 0-1范围映射到0-0.7，1-4范围映射到0.7-1.0
        # 这样0-1范围占用70%的输出空间，具有更高的分辨率
        threshold = 1.0
        near_range_ratio = 0.7  # 近距离范围占用的输出空间比例
        
        mapped_distances = torch.where(
            distances <= threshold,
            # 近距离：线性映射到 [0, near_range_ratio]
            distances * (near_range_ratio / threshold),
            # 远距离：线性映射到 [near_range_ratio, 1.0]
            near_range_ratio + (distances - threshold) * ((1.0 - near_range_ratio) / (max_range - threshold))
        )
        
        # 策略2：对数映射 (可选，取消注释使用)
        # log_base = 2.0
        # mapped_distances = torch.log(distances + 1.0) / torch.log(torch.tensor(max_range + 1.0))
        
        # 策略3：指数衰减映射 (可选，取消注释使用)
        # decay_factor = 2.0
        # mapped_distances = 1.0 - torch.exp(-decay_factor * distances / max_range)
        
        # 策略4：平方根映射 (可选，取消注释使用)
        # mapped_distances = torch.sqrt(distances / max_range)
        
        return torch.clamp(mapped_distances, 0.0, 1.0)
    
    def _update_command_resample(self):
        """更新命令重新采样逻辑，每1-10秒随机重新采样命令"""
        # 更新命令计时器
        self._command_timer += self.step_dt
        
        # 检查哪些环境需要重新采样命令
        resample_mask = self._command_timer >= self._command_timeout
        
        if torch.any(resample_mask):
            # 为需要重新采样的环境生成新命令
            resample_env_ids = torch.where(resample_mask)[0]
            
            # X轴: -1 到 1
            self._commands[resample_env_ids, 0] = torch.zeros_like(self._commands[resample_env_ids, 0]).uniform_(-1, 1)
            # Y轴: -1 到 1
            self._commands[resample_env_ids, 1] = torch.zeros_like(self._commands[resample_env_ids, 1]).uniform_(-1, 1)
            # Z轴(角速度): -1.0 到 1.0
            self._commands[resample_env_ids, 2] = torch.zeros_like(self._commands[resample_env_ids, 2]).uniform_(-1.0, 1.0)
            
            # 重置这些环境的命令计时器
            self._command_timer[resample_env_ids] = 0.0
            
            # 为这些环境设置新的重新采样间隔时间（1-10秒随机）
            self._command_timeout[resample_env_ids] = torch.zeros_like(self._command_timeout[resample_env_ids]).uniform_(
                self._command_resample_time_min, self._command_resample_time_max
            )
    
    def _compute_joint_vel_penalty(self, joint_ids: list[int] | None = None) -> torch.Tensor:
        """使用L2平方核惩罚关节速度
        
        Args:
            joint_ids: 指定的关节索引列表，如果为None则惩罚所有关节
            
        Returns:
            torch.Tensor: 关节速度惩罚值
        """
        joint_velocities = self._robot.data.joint_vel
        
        if joint_ids is not None:
            # 只惩罚指定的关节
            joint_velocities = joint_velocities[:, joint_ids]
            
        return torch.sum(torch.square(joint_velocities), dim=1)
    
    def _compute_stand_still_reward(self) -> torch.Tensor:
        """计算停止时静止奖励：当命令速度小于0.1且四只脚都在地面时给予奖励
        
        Returns:
            torch.Tensor: 静止奖励值
        """
        # 检查命令速度是否足够小（线性速度和角速度都要考虑）
        linear_cmd_norm = torch.norm(self._commands[:, :2], dim=1)  # 线性速度命令的模长
        angular_cmd_abs = torch.abs(self._commands[:, 2])  # 角速度命令的绝对值
        
        # 速度阈值：线性速度 < 0.1 且角速度 < 0.1
        low_speed_mask = (linear_cmd_norm < 0.1) & (angular_cmd_abs < 0.1)
        
        # 检查四只脚是否都在地面上
        # 使用接触力判断脚是否接触地面
        if self._contact_sensor.data.net_forces_w_history is not None:
            net_contact_forces = self._contact_sensor.data.net_forces_w_history
            feet_contact_forces = net_contact_forces[:, -1, self._feet_ids]  # 获取最新时刻的脚部接触力
            
            # 判断每只脚是否接触地面（接触力大于阈值）
            contact_threshold = 5.0  # 接触力阈值（N）
            feet_in_contact = torch.norm(feet_contact_forces, dim=-1) > contact_threshold  # [N, 4]
            
            # 检查是否四只脚都接触地面
            all_feet_contact = torch.all(feet_in_contact, dim=1)  # [N]
        else:
            # 如果没有接触力数据，则不给奖励
            all_feet_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 只有当速度命令小且四脚都接触地面时才给奖励
        stand_still_condition = low_speed_mask & all_feet_contact
        
        # 计算额外的稳定性奖励
        # 1. 机器人的实际速度也应该很小
        actual_linear_vel = torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1)
        actual_angular_vel = torch.abs(self._robot.data.root_ang_vel_b[:, 2])
        low_actual_speed = (actual_linear_vel < 0.2) & (actual_angular_vel < 0.2)
        
        # 2. 机器人姿态保持直立
        upright_condition = self._robot.data.projected_gravity_b[:, 2] < -0.8  # 接近-1表示直立
        
        # 综合条件：命令速度小 + 四脚接触 + 实际速度小 + 姿态直立
        final_condition = stand_still_condition & low_actual_speed & upright_condition
        
        # 返回奖励值（满足条件为1.0，否则为0.0）
        return final_condition.float()
