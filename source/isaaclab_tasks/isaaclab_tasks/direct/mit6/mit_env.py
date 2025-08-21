# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
from typing import Sequence

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
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
        # 新增：用于计算动作加速度惩罚的变量
        self._last_last_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity, yaw angular velocity, and height commands
        self._commands = torch.zeros(self.num_envs, 4, device=self.device)

        # 随机化默认关节位置相关变量
        self._randomized_default_joint_pos = torch.zeros(self.num_envs, 12, device=self.device)
        self._joint_pos_randomization_range = 0.03  # 关节位置随机化范围（弧度）


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
                 "long_contact_penalty",  # 新增：长期触地惩罚
                 "joint_direction_change",  # 新增：关节运动方向变化惩罚
                 "joint_pos_deviation",  # 新增：关节位置偏移惩罚
                 "action_acceleration_l2",  # 新增：动作加速度惩罚
                "joint_power_penalty",  # 新增：关节功率惩罚
                "feet_slip_penalty",  # 新增：脚滑移惩罚
                "feet_clearance_reward",  # 新增：脚离地高度奖励
                "feet_height_reward",  # 新增：摆动脚高度奖励
                "stand_still_penalty",  # 新增：站立静止惩罚
                "height_tracking",  # 新增：高度跟踪奖励

            ]
        }
        self._fallen_timeout =2.5  # 倒地超过3秒触发重置
        self._fallen_timer = torch.zeros(self.num_envs, device=self.device)  # 跟踪每个环境倒地的持续时间
        
        # 长期触地惩罚相关变量
        self._feet_contact_timer = torch.zeros(self.num_envs, 4, device=self.device)  # 跟踪每只脚的接触时间 [N, 4]
        self._long_contact_threshold = 3.0  # 长期接触阈值：1秒
        
        self._previous_joint_vel = torch.zeros(self.num_envs, 12, device=self.device)  # 存储上一时刻的关节速度

        # 命令重新采样相关变量
        self._command_resample_time_min = 3  # 最小重新采样间隔（秒）
        self._command_resample_time_max = 10.0  # 最大重新采样间隔（秒）
        self._command_timer = torch.zeros(self.num_envs, device=self.device)  # 命令计时器
        self._command_timeout = torch.zeros(self.num_envs, device=self.device)  # 每个环境的命令超时时间
        
        # Swing peak tracking for feet height reward (JAX style)
        self._swing_peak = torch.full((self.num_envs, 4), -0.25, device=self.device)  # 跟踪每只脚的摆动峰值高度，初始化为默认脚部偏移 [N, 4]
        self._last_contact = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.bool)  # 上一时刻的接触状态
        
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*_foot")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies([".*_thigh",".*_calf"]) #, "base"
        self._front_calf_ids, _ = self._contact_sensor.find_bodies("F[LR]_calf")  # 匹配FL_calf和FR_calf
        self._thigh_ids, _ = self._contact_sensor.find_bodies(".*_thigh")  # 获取大腿body的索引
        
        # 用于获取脚部位置的body索引（需要在_setup_scene后初始化）
        self._feet_body_ids =  self._robot.find_bodies(".*_foot")[0]  # 获取所有脚部body的索引
        print(f"脚部body索引: {self._feet_body_ids}")

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

        #全为10，调试用
        # self._actions = torch.full_like(self._actions, 10.0, device=self.device)
        #把cfg.action_scale列表转为 torch.Tensor
        action_scales = torch.tensor(self.cfg.action_scale, device=self.device).view(1, -1)
        
        # 每个时间步都有30%的概率将机器人设置为walking姿态
        
        #self._processed_actions = action_scales * self._actions + self._robot.data.default_joint_pos
        self._processed_actions = action_scales * self._actions + self._randomized_default_joint_pos
        # 更新命令重新采样
        self._update_command_resample()

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        # Update action history for acceleration penalty
        self._last_last_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()
        
        # Add noise to observations
        device = self._robot.device
        
        # Commands with height (no noise)
        velocity_commands = self._commands[:, :3]  # [3] - x_vel, y_vel, yaw_vel
        height_command = self._commands[:, 3:4]    # [1] - target height

        # IMU Angular Velocity (with noise ±0.1)
        imu_ang_vel = self._imu_sensor.data.ang_vel_b  # [3]
        ang_vel_noise = torch.empty_like(imu_ang_vel).uniform_(-0.05, 0.05)
        imu_ang_vel = imu_ang_vel + ang_vel_noise  

        # # IMU Orientation (with noise ±0.05)
        imu_lin_acc = self._imu_sensor.data.lin_acc_b
        orientation_noise = torch.empty_like(imu_lin_acc).uniform_(-0.05, 0.05)
        imu_lin_acc = imu_lin_acc + orientation_noise

        #base velocity
        #imu_lin_acc = self._robot.data.root_lin_vel_b[:, :3]  # [N, 3] - 获取机器人基座的线速度
        # Projected Gravity (with noise ±0.05)

        proj_gravity = self._robot.data.projected_gravity_b  # [3]
        gravity_noise = torch.empty_like(proj_gravity).uniform_(-0.05, 0.05)
        proj_gravity = proj_gravity + gravity_noise

        joint_positions = self._robot.data.joint_pos - self._randomized_default_joint_pos  # [12]
        joint_pos_noise = torch.empty_like(joint_positions).uniform_(-0.03, 0.03)
        joint_positions = joint_positions + joint_pos_noise

        # Joint Velocities (with noise ±1.5)
        joint_velocities = self._robot.data.joint_vel  # [12]
        joint_vel_noise = torch.empty_like(joint_velocities).uniform_(-0.5, 0.5)
        joint_velocities = joint_velocities + joint_vel_noise

        # Last Actions (no noise)
        actions = self._actions  # [12]

        # Foot height information (for critic network)
        feet_pos_w = self._robot.data.body_pos_w[:, self._feet_body_ids]  # [N, 4, 3]
        robot_height = self._robot.data.root_pos_w[:, 2:3]  # [N, 1]
        
        # Calculate relative foot heights (relative to robot body)
        feet_relative_heights = feet_pos_w[:, :, 2] - robot_height  # [N, 4] - 脚部相对于身体的高度
        feet_heights = feet_relative_heights  # [4] - 四只脚的相对高度

        height_data = None
        if isinstance(self.cfg, MitRoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
        obs = torch.cat(
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
                    #height_data,         # 
                    #lidar_depth_data,   # [64] - 深度投影信息
                )
                if tensor is not None
            ],
            dim=-1,
        )
        critic = torch.cat(
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
                    feet_heights.flatten(start_dim=1),  # [4] - 四只脚的相对高度信息
                    height_data,         # 
                    #lidar_depth_data,   # [64] - 深度投影信息
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs, "critic": critic}
        #observations = {"policy": teacher}
        #observations = {"policy": obs }
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
        # feet air time (JAX style implementation)
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        
        # JAX style: (air_time - 0.1) * first_contact, only reward when moving
        has_movement_command = torch.norm(self._commands[:, :2], dim=1) > 0.1
        air_time = torch.sum((last_air_time - 0.3) * first_contact, dim=1) * has_movement_command.float()


        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        #upward_reward = self._compute_upward_reward()
        
        # joint position deviation penalty
        joint_pos_deviation = self._compute_joint_pos_deviation_penalty()
        
        # action acceleration penalty
        #action_acceleration_penalty = self._compute_action_acceleration_penalty()

        #power penalty
        #joint_power_penalty = self._compute_joint_power_penalty()
        
        # slip penalty
        #feet_slip_penalty = self._compute_feet_slip_penalty()
        
        # feet clearance reward
        #feet_clearance_reward = self._compute_feet_clearance_reward()
        
        # feet height reward
        #feet_height_reward = self._compute_feet_height_reward()
        
        # stand still penalty
        stand_still_penalty = self._compute_stand_still_penalty()
        
        # height tracking reward
        height_tracking_reward = self._compute_height_tracking_reward()
        
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
            # "upward_reward": upward_reward * self.cfg.upward_reward_scale * self.step_dt,
            "joint_pos_deviation": joint_pos_deviation * self.cfg.joint_pos_deviation_scale * self.step_dt,  # 关节位置偏移惩罚
            # "action_acceleration_l2": action_acceleration_penalty * self.cfg.action_acceleration_reward_scale * self.step_dt,  # 动作加速度惩罚
            # "joint_power_penalty": joint_power_penalty * self.cfg.joint_power_reward_scale * self.step_dt,  # 关节功率惩罚
            # "feet_slip_penalty": feet_slip_penalty * self.cfg.feet_slip_penalty_scale * self.step_dt,  # 脚滑移惩罚
            # "feet_clearance_reward": feet_clearance_reward * self.cfg.feet_clearance_reward_scale * self.step_dt,  # 脚离地高度奖励
            # "feet_height_reward": feet_height_reward * self.cfg.feet_height_reward_scale * self.step_dt,  # 摆动脚高度奖励
            "stand_still_penalty": stand_still_penalty * self.cfg.stand_still_penalty_scale * self.step_dt,  # 站立静止惩罚
            "height_tracking": height_tracking_reward * self.cfg.height_tracking_scale * self.step_dt,  # 高度跟踪奖励
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward


    def _compute_joint_power_penalty(self) -> torch.Tensor:
        """计算关节功率惩罚
        
        参考能量消耗计算：功率 = |速度| × |扭矩|
        这个惩罚鼓励机器人采用更节能的控制策略
        
        Returns:
            torch.Tensor: 关节功率惩罚值 [N]
        """
        # 参考 _cost_energy 方法：使用速度和扭矩的绝对值乘积
        # power_penalty = sum(|qvel| * |qfrc_actuator|)
        power_penalty = torch.sum(
            torch.abs(self._robot.data.joint_vel) * torch.abs(self._robot.data.applied_torque), 
            dim=1
        )
        
        return power_penalty

    def _compute_joint_pos_deviation_penalty(self) -> torch.Tensor:
            """计算关节位置偏离随机化默认位置的惩罚
            
            Returns:
                torch.Tensor: 关节位置偏移惩罚值
            """
            # 获取当前关节位置和随机化的默认关节位置
            current_joint_pos = self._robot.data.joint_pos  # [N, 12]
            randomized_default_joint_pos = self._randomized_default_joint_pos  # [N, 12]
            
            # 计算偏移量
            joint_pos_deviation = current_joint_pos - randomized_default_joint_pos  # [N, 12]
            
            # 使用L2范数计算偏移惩罚（平方和）
            deviation_penalty = torch.sum(torch.square(joint_pos_deviation), dim=1)  # [N]
            
            return deviation_penalty
    def _compute_joint_direction_change_penalty(self) -> torch.Tensor:
            """计算关节运动方向变化惩罚：检测关节速度方向是否与上次相反
            返回0（无方向反转）或1（有方向反转）
            
            Returns:
                torch.Tensor: 关节方向反转惩罚值，0或1
            """
            current_joint_vel = self._robot.data.joint_vel  # [N, 12]
            
            # 速度阈值：只对有显著运动的关节检查方向反转
            velocity_threshold = 0.5
            
            # 检查当前和之前的速度是否都足够大
            current_significant = torch.abs(current_joint_vel) > velocity_threshold  # [N, 12]
            previous_significant = torch.abs(self._previous_joint_vel) > velocity_threshold  # [N, 12]
            
            # 只对两个时刻都有显著运动的关节检查方向反转
            check_mask = current_significant & previous_significant  # [N, 12]
            
            # 检查方向是否相反：符号相反表示方向反转
            direction_opposite = (current_joint_vel * self._previous_joint_vel < 0) & check_mask  # [N, 12]
            self._previous_joint_vel = current_joint_vel.clone()  # 更新上一时刻的关节速度
            # 统计每个环境中有多少个关节发生了方向反转
            direction_change_count = torch.sum(direction_opposite, dim=1)  # [N]
            return direction_change_count.float()  # 返回0或1，表示是否有方向反转

    def _compute_action_acceleration_penalty(self) -> torch.Tensor:
        """计算动作加速度惩罚
        
        计算公式: torch.sum(torch.square(self.actions - 2 * self.last_actions + self.last_last_actions), dim=1)
        这个公式计算的是动作的二阶差分（加速度）的L2范数
        
        Returns:
            torch.Tensor: 动作加速度惩罚值 [N]
        """
        # 计算动作加速度: a_t - 2*a_{t-1} + a_{t-2}
        action_acceleration = self._actions - 2 * self._previous_actions + self._last_last_actions
        
        # 计算L2范数的平方（惩罚值）
        acceleration_penalty = torch.sum(torch.square(action_acceleration), dim=1)
        
        return acceleration_penalty

    def _compute_feet_slip_penalty(self) -> torch.Tensor:
        """计算脚滑移惩罚 (JAX style implementation)
        
        参考 JAX 实现：
        cmd_norm = jp.linalg.norm(info["command"])
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
        return jp.sum(vel_xy_norm_sq * contact) * (cmd_norm > 0.01)
        
        Returns:
            torch.Tensor: 脚滑移惩罚值 [N]
        """
        # 计算命令速度的模长
        cmd_norm = torch.norm(self._commands[:, :2], dim=1)  # 只考虑 x, y 线性速度命令
        
        # 获取脚部接触状态
        contact = torch.zeros(self.num_envs, 4, device=self.device)
        if self._contact_sensor.data.net_forces_w_history is not None:
            net_contact_forces = self._contact_sensor.data.net_forces_w_history
            feet_contact_forces = net_contact_forces[:, -1, self._feet_ids]  # [N, 4, 3]
            contact = (torch.norm(feet_contact_forces, dim=-1) > 1.0).float()  # [N, 4]
        
        # 获取脚部在世界坐标系下的线性速度 [N, 4, 3]
        feet_vel_w = self._robot.data.body_lin_vel_w[:, self._feet_body_ids]  # [N, 4, 3]
        
        # 获取机器人根部速度作为参考
        robot_vel_w = self._robot.data.root_lin_vel_w[:, None, :]  # [N, 1, 3]
        
        # 计算脚部相对于机器人身体的相对速度
        # 这样在机器人整体运动时，只关注脚部的相对滑移
        feet_relative_vel = feet_vel_w - robot_vel_w  # [N, 4, 3]
        feet_vel_xy = feet_relative_vel[:, :, :2]  # [N, 4, 2] - 只取xy平面相对速度
        
        # 计算每只脚 xy 平面相对速度的平方
        vel_xy_norm_sq = torch.sum(torch.square(feet_vel_xy), dim=-1)  # [N, 4]
        
        # 计算滑移惩罚：只有在接触地面时才计算滑移
        slip_penalty = torch.sum(vel_xy_norm_sq * contact, dim=1)  # [N]
        
        # 只有在有运动命令时才给惩罚
        has_movement_command = (cmd_norm > 0.05).float()
        
        return slip_penalty * has_movement_command

    def _compute_feet_clearance_reward(self) -> torch.Tensor:
        """计算脚离地高度奖励 (JAX style implementation)
        
        参考 JAX 实现：
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
        foot_pos = data.site_xpos[self._feet_site_id]
        foot_z = foot_pos[..., -1]
        delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
        return jp.sum(delta * vel_norm)
        
        Returns:
            torch.Tensor: 脚离地高度奖励值 [N]
        """
        # 获取脚部在世界坐标系下的位置 [N, num_feet, 3]
        feet_pos_w = self._robot.data.body_pos_w[:, self._feet_body_ids]  # [N, 4, 3]
        feet_z = feet_pos_w[:, :, 2]  # [N, 4] - 脚部z坐标（高度）
        
        # 获取机器人根部高度作为参考
        robot_height = self._robot.data.root_pos_w[:, 2:3]  # [N, 1]
        
        # 计算脚部相对于机器人身体的相对高度
        # 在崎岖地形中，相对高度比绝对高度更有意义
        feet_relative_height = feet_z - robot_height  # [N, 4] - 脚部相对于身体的高度
        
        # 获取脚部在世界坐标系下的速度
        # 使用脚部body的线性速度
        feet_vel_w = self._robot.data.body_lin_vel_w[:, self._feet_body_ids]  # [N, 4, 3]
        feet_vel_xy = feet_vel_w[:, :, :2]  # [N, 4, 2] - xy平面速度
        
        # 计算每只脚的xy平面速度模长
        vel_norm = torch.sqrt(torch.norm(feet_vel_xy, dim=-1))  # [N, 4]
        
        # 设定最大脚部相对高度阈值
        max_foot_height = 0.1 - 0.25
        
        # 计算脚部相对高度与最大高度的差值
        delta = torch.abs(feet_relative_height - max_foot_height)  # [N, 4]
        
        # 返回高度差值与速度的乘积，对所有脚求和
        clearance_reward = torch.sum(delta * vel_norm, dim=1)  # [N]
        
        return clearance_reward

    def _compute_feet_height_reward(self) -> torch.Tensor:
        """计算摆动脚高度奖励 (JAX style implementation)
        
        参考 JAX 实现：
        cmd_norm = jp.linalg.norm(info["command"])
        error = swing_peak / self._config.reward_config.max_foot_height - 1.0
        return jp.sum(jp.square(error) * first_contact) * (cmd_norm > 0.01)
        
        JAX代码中的swing peak跟踪：
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)
        
        Returns:
            torch.Tensor: 摆动脚高度奖励值 [N]
        """
        # 获取当前接触状态
        contact = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.bool)
        if self._contact_sensor.data.net_forces_w_history is not None:
            net_contact_forces = self._contact_sensor.data.net_forces_w_history
            feet_contact_forces = net_contact_forces[:, -1, self._feet_ids]  # [N, 4, 3]
            contact = (torch.norm(feet_contact_forces, dim=-1) > 1.0)  # [N, 4]
        
        # 获取脚部的真实位置 [N, 4, 3]
        feet_pos_w = self._robot.data.body_pos_w[:, self._feet_body_ids]  # [N, 4, 3]
        feet_z = feet_pos_w[:, :, 2]  # [N, 4] - 脚部z坐标（高度）
        
        # 获取机器人根部高度作为参考
        robot_height = self._robot.data.root_pos_w[:, 2:3]  # [N, 1]
        
        # 计算脚部相对于机器人身体的相对高度
        # 在崎岖地形中，相对高度比绝对高度更有意义
        feet_relative_height = feet_z - robot_height  # [N, 4] - 脚部相对于身体的高度
        
        # 更新swing peak：跟踪每只脚在空中时的最大相对高度
        # 当脚接触地面时重置swing peak，空中时记录最大相对高度
        self._swing_peak = torch.where(
            contact,
            torch.full_like(self._swing_peak, -0.25),  # 接触时重置为-0.25（脚部相对于身体的默认偏移）
            torch.maximum(self._swing_peak, feet_relative_height)  # 空中时记录最大相对高度
        )
        
        # 获取首次接触状态
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]  # [N, 4]
        
        # 计算命令速度的模长
        cmd_norm = torch.norm(self._commands[:, :2], dim=1)  # [N]
        
        # 设定最大脚部相对高度
        
        # 计算归一化误差：swing_peak / max_foot_height - 1.0
        #error = self._swing_peak / max_foot_height - 1.0  # [N, 4]
        error = (self._swing_peak+0.25) / 0.1 - 1.0  # 修正：加上脚部默认偏移
        # 计算平方误差与首次接触的乘积
        height_error = torch.sum(torch.square(error) * first_contact, dim=1)  # [N]
        
        # 只有在有运动命令时才给奖励
        has_movement_command = (cmd_norm > 0.05).float()
        
        return height_error * has_movement_command



    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        
        # Check body collision (base contact)
        body_collision = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        
        # Check thigh collision (new addition)
        # thigh_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # if hasattr(self, '_thigh_ids') and len(self._thigh_ids) > 0:
        #     thigh_contact_forces = net_contact_forces[:, :, self._thigh_ids]  # [N, T, num_thighs]
        #     thigh_contact_norm = torch.norm(thigh_contact_forces, dim=-1)  # [N, T, num_thighs]
        #     thigh_max_contact = torch.max(thigh_contact_norm, dim=1)[0]  # [N, num_thighs]
        #     thigh_collision = torch.any(thigh_max_contact > 1.0, dim=1)  # [N]
        
        # Combine collision conditions: base OR thigh contact
        collision = body_collision #| thigh_collision
        
        return collision, time_out

    # def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    #     """只保留超时重置，移除碰撞重置"""
    #     # 保留原来的超时逻辑
    #     time_out = self.episode_length_buf >= self.max_episode_length - 1
        
    #     # 移除碰撞检测，永远不因为碰撞重置
    #     died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
    #     return died, time_out

    # def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    #     """检测身体碰撞或两只前小腿同时碰撞时重置"""
    #     # 保留原来的超时逻辑
    #     time_out = self.episode_length_buf >= self.max_episode_length - 1
        
    #     # 获取接触力数据
    #     net_contact_forces = self._contact_sensor.data.net_forces_w_history
    #     died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
    #     if net_contact_forces is not None:
    #         # 1. 检查身体碰撞
    #         base_contact_forces = net_contact_forces[:, -1, self._base_id]  # [N, 1]
    #         base_collision = torch.norm(base_contact_forces, dim=-1) > 1.0  # [N]
    #         if base_contact_forces.dim() > 1:
    #             base_collision = torch.any(base_collision, dim=1)
            
    #         # 2. 检查前小腿碰撞（需要两只前小腿同时碰撞）
    #         front_calf_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            
    #         if hasattr(self, '_front_calf_ids') and len(self._front_calf_ids) >= 2:
    #             front_calf_forces = net_contact_forces[:, -1, self._front_calf_ids]  # [N, 2]
                
    #             # 碰撞阈值
    #             collision_threshold = 1.0
                
    #             # 检查每只前小腿是否碰撞
    #             front_calf_contact = torch.norm(front_calf_forces, dim=-1) > collision_threshold  # [N, 2]
                
    #             # 只有当两只前小腿都碰撞时才重置
    #             front_calf_collision = torch.all(front_calf_contact, dim=1)  # [N]
            
    #         # 组合重置条件：身体碰撞 OR 两只前小腿同时碰撞
    #         died = base_collision | front_calf_collision
        
    #     return died, time_out

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
        threshold_deg = 30
        
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
    
    def _compute_long_contact_penalty(self) -> torch.Tensor:
        """计算长期触地惩罚：当命令速度大于0.1且脚接触地面超过1秒时给予惩罚
        
        Returns:
            torch.Tensor: 长期触地惩罚值
        """
        # 检查命令速度是否大于阈值
        linear_cmd_norm = torch.norm(self._commands[:, :2], dim=1)  # 线性速度命令的模长
        angular_cmd_abs = torch.abs(self._commands[:, 2])  # 角速度命令的绝对值
        
        # 速度阈值：线性速度 > 0.1 或角速度 > 0.1
        high_speed_mask = (linear_cmd_norm > 0.1) | (angular_cmd_abs > 0.1)
        
        # 检查脚部接触状态
        if self._contact_sensor.data.net_forces_w_history is not None:
            net_contact_forces = self._contact_sensor.data.net_forces_w_history
            feet_contact_forces = net_contact_forces[:, -1, self._feet_ids]  # 获取最新时刻的脚部接触力
            
            # 判断每只脚是否接触地面（接触力大于阈值）
            contact_threshold = 1.0  # 接触力阈值（N）
            feet_in_contact = torch.norm(feet_contact_forces, dim=-1) > contact_threshold  # [N, 4]
            
            # 更新接触计时器
            # 如果脚接触地面，累加时间；否则重置计时器
            self._feet_contact_timer = torch.where(
                feet_in_contact,
                self._feet_contact_timer + self.step_dt,  # 累加接触时间
                torch.zeros_like(self._feet_contact_timer)  # 重置计时器
            )
            
            # 检查是否有脚接触时间超过阈值
            long_contact_mask = self._feet_contact_timer > self._long_contact_threshold  # [N, 4]
            
            # 计算每个环境中长期接触的脚数量
            long_contact_count = torch.sum(long_contact_mask, dim=1)  # [N]
            
        else:
            # 如果没有接触力数据，则不给惩罚
            long_contact_count = torch.zeros(self.num_envs, device=self.device)
        
        # 只有当速度命令大且有脚长期接触时才给惩罚
        penalty_condition = high_speed_mask.float() * long_contact_count.float()
        
        # 返回惩罚值（越多脚长期接触惩罚越大）
        return penalty_condition
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        
        # 随机化默认关节位置
        self._randomize_default_joint_positions(env_ids)
        
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._last_last_actions[env_ids] = 0.0
        self._fallen_timer[env_ids] = 0.0
        # 重置长期触地计时器
        self._feet_contact_timer[env_ids] = 0.0
        # 重置swing peak跟踪为默认脚部偏移值
        self._swing_peak[env_ids] = -0.1
        self._last_contact[env_ids] = False
        # 重置命令计时器和超时时间
        self._command_timer[env_ids] = 0.0
        # 为重置的环境设置新的重新采样间隔时间（1-10秒随机）
        self._command_timeout[env_ids] = torch.zeros_like(self._command_timeout[env_ids]).uniform_(
            self._command_resample_time_min, self._command_resample_time_max
        )
        
        # 只在初始化时采样一次命令，后续由_update_command_resample自动处理
        # X轴: -1 到 1
        self._commands[env_ids, 0] = torch.zeros_like(self._commands[env_ids, 0]).uniform_(-1, 1)
        # # Y轴: -1 到 1
        self._commands[env_ids, 1] = torch.zeros_like(self._commands[env_ids, 1]).uniform_(-1, 1)
        # # Z轴(角速度): -1.0 到 1.0
        self._commands[env_ids, 2] = torch.zeros_like(self._commands[env_ids, 2]).uniform_(-1, 1)
        #        # 高度目标: 0.0 到 0.4
        self._commands[env_ids, 3] = torch.zeros_like(self._commands[env_ids, 3]).uniform_(0.15, 0.4)
        
        # 随机化机器人根状态（位置、方向、速度）和关节状态
       # self._randomize_root_state(env_ids)
        # Logging
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


    def _randomize_default_joint_positions(self, env_ids: torch.Tensor):
        """随机化指定环境的默认关节位置
        
        Args:
            env_ids: 需要随机化的环境ID
        """
        # 获取原始默认关节位置
        original_default_pos = self._robot.data.default_joint_pos[env_ids]  # [N, 12]
        
        # 生成随机偏移量 [-range, +range]
        random_offset = torch.empty_like(original_default_pos).uniform_(
            -self._joint_pos_randomization_range, 
            self._joint_pos_randomization_range
        )
        
        # 应用随机偏移
        self._randomized_default_joint_pos[env_ids] = original_default_pos + random_offset

    def _randomize_root_state(self, env_ids: torch.Tensor):
        """随机化机器人根状态（位置、方向和速度）
        
        基于 reset_root_state_uniform 函数实现，随机化根位置、方向和速度
        
        Args:
            env_ids: 需要随机化的环境ID
        """
        # 姿态随机化范围
        pose_range = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.0, 0.0),  # 保持Z轴不变
            "roll":  (-3.14, 3.14),
            "pitch": (-3.14, 3.14),
           # "yaw": (-3.14, 3.14),
        }
        
        # 速度随机化范围
        velocity_range = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.5, 0.5),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5),
            "yaw": (-0.5, 0.5),
        }
        
        # 获取默认根状态
        root_states = self._robot.data.default_root_state[env_ids].clone()
        
        # 随机化姿态
        pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        pose_ranges_list = [pose_range.get(key, (0.0, 0.0)) for key in pose_keys]
        pose_ranges = torch.tensor(pose_ranges_list, device=self.device)
        pose_rand_samples = math_utils.sample_uniform(
            pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        
        # 计算新位置和方向
        positions = root_states[:, 0:3] + self._terrain.env_origins[env_ids] + pose_rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(
            pose_rand_samples[:, 3], pose_rand_samples[:, 4], pose_rand_samples[:, 5]
        )
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
        
        # 随机化速度
        vel_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        vel_ranges_list = [velocity_range.get(key, (0.0, 0.0)) for key in vel_keys]
        vel_ranges = torch.tensor(vel_ranges_list, device=self.device)
        vel_rand_samples = math_utils.sample_uniform(
            vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        
        velocities = root_states[:, 7:13] + vel_rand_samples
        
        # 设置到物理仿真中
        env_ids_list = env_ids#.tolist()
        self._robot.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids_list)
        self._robot.write_root_velocity_to_sim(velocities, env_ids_list)
        
        # 设置关节状态
        joint_pos = self._randomized_default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids_list)

    def _update_command_resample(self):
        """更新命令重新采样逻辑，每1-10秒随机重新采样命令
        前3个命令（X、Y、Z轴）会额外增加采样0值的概率
        """
        # 更新命令计时器
        self._command_timer += self.step_dt
        
        # 检查哪些环境需要重新采样命令
        resample_mask = self._command_timer >= self._command_timeout
        
        if torch.any(resample_mask):
            # 为需要重新采样的环境生成新命令
            resample_env_ids = torch.where(resample_mask)[0]
            
            # 为前3个命令增加采样0值的概率
            zero_prob = 0.05# 30%的概率采样0值
            
            # X轴: -1 到 1，但有额外概率采样0
            if torch.rand(1).item() < zero_prob:
                self._commands[resample_env_ids, 0] = 0.0
            else:
                self._commands[resample_env_ids, 0] = torch.zeros_like(self._commands[resample_env_ids, 0]).uniform_(-1, 1)
            
            # Y轴: -1 到 1，但有额外概率采样0
            if torch.rand(1).item() < zero_prob:
                self._commands[resample_env_ids, 1] = 0.0
            else:
                self._commands[resample_env_ids, 1] = torch.zeros_like(self._commands[resample_env_ids, 1]).uniform_(-1, 1)
            
            # Z轴(角速度): -1.0 到 1.0，但有额外概率采样0
            if torch.rand(1).item() < zero_prob:
                self._commands[resample_env_ids, 2] = 0.0
            else:
                self._commands[resample_env_ids, 2] = torch.zeros_like(self._commands[resample_env_ids, 2]).uniform_(-1, 1)
            
            # 高度目标: 0.0 到 0.4 (保持原样，不增加0值概率)
            self._commands[resample_env_ids, 3] = torch.zeros_like(self._commands[resample_env_ids, 3]).uniform_(0.15, 0.4)
            
            # 重置这些环境的命令计时器
            self._command_timer[resample_env_ids] = 0.0
    def _compute_height_tracking_reward(self) -> torch.Tensor:
        """计算高度跟踪奖励
        
        跟踪命令中指定的目标高度，使用指数函数进行奖励映射
        当机器人高度接近目标高度时给予正奖励
        
        Returns:
            torch.Tensor: 高度跟踪奖励值 [N]
        """
        # 获取机器人当前高度
        current_height = self._robot.data.root_pos_w[:, 2]  # [N]
        
        # 获取目标高度命令
        target_height = self._commands[:, 3]  # [N] - 第4个命令是高度目标
        
        # 计算高度误差
        height_error = torch.square(current_height - target_height)  # [N]
        
        # 使用指数函数映射奖励，误差越小奖励越高
        # 参数0.25可以调整奖励的敏感度
        height_tracking_reward = torch.exp(-height_error / 0.02)  # [N]
        
        return height_tracking_reward

    def _compute_stand_still_penalty(self) -> torch.Tensor:
        """计算站立静止惩罚 (JAX style implementation)
        
        参考 JAX 实现：
        cmd_norm = jp.linalg.norm(commands)
        return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.01)
        
        当命令速度很小时，惩罚关节位置偏离默认姿态，鼓励机器人在静止时保持默认姿态
        
        Returns:
            torch.Tensor: 站立静止惩罚值 [N]
        """
        # 计算命令速度的模长（包含x, y线性速度和z角速度命令）
        linear_cmd_norm = torch.norm(self._commands[:, :2], dim=1)  # [N] - x, y线性速度
        angular_cmd_abs = torch.abs(self._commands[:, 2])  # [N] - z角速度（绝对值）
        
        # 获取当前关节位置和随机化的默认关节位置
        current_joint_pos = self._robot.data.joint_pos  # [N, 12]
        default_joint_pos = self._randomized_default_joint_pos  # [N, 12]
        
        # 计算关节位置偏离默认姿态的绝对值之和
        joint_deviation = torch.sum(torch.abs(current_joint_pos - default_joint_pos), dim=1)  # [N]
        
        # 只有在命令速度很小时才给惩罚（线性速度 < 0.05 且角速度 < 0.05）
        is_stand_still = ((linear_cmd_norm < 0.05) & (angular_cmd_abs < 0.05)).float()  # [N]
        
        return joint_deviation * is_stand_still