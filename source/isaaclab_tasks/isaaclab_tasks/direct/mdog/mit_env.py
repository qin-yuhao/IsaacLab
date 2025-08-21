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
        # 新增：用于计算动作加速度惩罚的变量
        self._last_last_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity, yaw angular velocity, and height commands
        self._commands = torch.zeros(self.num_envs, 4, device=self.device)

        # 随机化默认关节位置相关变量
        self._randomized_default_joint_pos = torch.zeros(self.num_envs, 12, device=self.device)
        self._joint_pos_randomization_range = 0.05  # 关节位置随机化范围（弧度）


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
                "track_height_exp",  # 新增：身体高度跟踪奖励
                "stand_still",  # 新增：站立静止惩罚
                "feet_stumble_penalty",  # 新增：脚绊倒惩罚
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
        
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*_foot")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies([".*_thigh", "base",".*_calf"]) #  ,
        self._front_calf_ids, _ = self._contact_sensor.find_bodies("F[LR]_calf")  # 匹配FL_calf和FR_calf
        self._thigh_ids, _ = self._contact_sensor.find_bodies(".*_thigh")  # 获取大腿body的索引

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
        action_scales = torch.tensor(self.cfg.action_scale, device=self.device).view(1, -1)
        
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
        ang_vel_noise = torch.empty_like(imu_ang_vel).uniform_(-0.1, 0.1)
        imu_ang_vel = imu_ang_vel + ang_vel_noise  

        # IMU Orientation (with noise ±0.05)
        imu_lin_acc = self._imu_sensor.data.lin_acc_b
        orientation_noise = torch.empty_like(imu_lin_acc).uniform_(-0.3, 0.3)
        imu_lin_acc = imu_lin_acc + orientation_noise

        # Projected Gravity (with noise ±0.05)
        proj_gravity = self._robot.data.projected_gravity_b  # [3]
        gravity_noise = torch.empty_like(proj_gravity).uniform_(-0.05, 0.05)
        proj_gravity = proj_gravity + gravity_noise

        # Joint Positions (with noise ±0.01)
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

        #base vel
        base_lin_vel = self._robot.data.root_lin_vel_b
        #base_ang_vel 
        base_ang_vel = self._robot.data.root_ang_vel_b
        


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
                    base_lin_vel,         # [3]
                    base_ang_vel,         # [3]
                    imu_ang_vel,         # [3]
                    imu_lin_acc,         # [3]
                    proj_gravity,        # [3]
                    joint_positions,     # [12]
                    joint_velocities,    # [12]
                    actions,            # [12]
                    height_data,         # 
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
        
        # body height tracking
        current_height = self._robot.data.root_pos_w[:, 2]
        height_error = torch.square(self._commands[:, 3] - current_height)
        height_error_mapped = torch.exp(-height_error / 0.15)
        
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
        # air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
        #     torch.norm(self._commands[:, :2], dim=1) > 0.1
        # )
        air_time = torch.sum(torch.abs(last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )  # 使用绝对值来计算脚离地时间
        #exp
        air_time = torch.square(air_time)  # 使用平方函数来计算脚离地时间奖励
        #air_time = torch.exp(-air_time / 0.2)  # 使用指数衰减函数来计算脚离地时间奖励
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        upward_reward = self._compute_upward_reward()
        

        long_contact_penalty = self._compute_long_contact_penalty()

        joint_direction_change = self._compute_joint_direction_change_penalty()
        
        # joint position deviation penalty
        joint_pos_deviation = self._compute_joint_pos_deviation_penalty()
        
        # action acceleration penalty
        action_acceleration_penalty = self._compute_action_acceleration_penalty()

        #power penalty
        joint_power_penalty = self._compute_joint_power_penalty()

        #stand still reward
        stand_still = self._compute_stand_still_penalty()

        feet_stumble_penalty = self._compute_feet_stumble_penalty()
        
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "track_height_exp": height_error_mapped * self.cfg.height_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "upward_reward": upward_reward * self.cfg.upward_reward_scale * self.step_dt,
            "long_contact_penalty": long_contact_penalty * self.cfg.long_contact_penalty_scale * self.step_dt,  # 长期触地惩罚
             "joint_direction_change": joint_direction_change * self.cfg.joint_direction_change_scale * self.step_dt,  # 关节运动方向变化惩罚
             "joint_pos_deviation": joint_pos_deviation * self.cfg.joint_pos_deviation_scale * self.step_dt,  # 关节位置偏移惩罚
             "action_acceleration_l2": action_acceleration_penalty * self.cfg.action_acceleration_reward_scale * self.step_dt,  # 动作加速度惩罚
            "joint_power_penalty": joint_power_penalty * self.cfg.joint_power_reward_scale * self.step_dt,  # 关节功率惩罚
            "stand_still": stand_still * self.cfg.stand_still * self.step_dt,  # 站立静止惩罚
            "feet_stumble_penalty": feet_stumble_penalty * self.cfg.feet_stumble_penalty_scale * self.step_dt,  # 脚绊倒惩罚
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward


    def _compute_stand_still_penalty(self) -> torch.Tensor:
            """计算站立静止惩罚 (JAX style implementation)
            
            参考 JAX 实现：
            cmd_norm = jp.linalg.norm(commands)
            return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.01)
            
            当命令速度很小时，惩罚关节位置偏离默认姿态，鼓励机器人在静止时保持默认姿态
            
            Returns:
                torch.Tensor: 站立静止惩罚值 [N]
            """
            # 计算命令速度的模长
            cmd_norm = torch.norm(self._commands[:, :2], dim=1)
            #wz
            cmd_wz = torch.abs(self._commands[:, 2])  # 角速度命令的绝对值


            
            # 获取当前关节位置和随机化的默认关节位置
            current_joint_pos = self._robot.data.joint_pos  # [N, 12]
            default_joint_pos = self._randomized_default_joint_pos  # [N, 12]
            
            # 计算关节位置偏离默认姿态的绝对值之和
            joint_deviation = torch.sum(torch.abs(current_joint_pos - default_joint_pos), dim=1)  # [N]
            
            # 在命令速度很小 而且 命令旋转速度也很小的情况下，给予惩罚
            is_stand_still = (cmd_norm < 0.05 ) & (cmd_wz < 0.05)  # [N]
            
            return joint_deviation * is_stand_still
    

    def _compute_joint_power_penalty(self) -> torch.Tensor:
        """计算关节功率惩罚
        
        功率 = 扭矩 × 角速度
        这个惩罚鼓励机器人采用更节能的控制策略
        
        Returns:
            torch.Tensor: 关节功率惩罚值 [N]
        """
        # 计算每个关节的功率：扭矩 × 角速度
        joint_power = self._robot.data.applied_torque * self._robot.data.joint_vel
        
        # 使用L1范数计算功率惩罚（也可以改为L2）
        power_penalty = torch.sum(torch.abs(joint_power), dim=1)
        
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

    # def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    #     """只保留超时重置，移除碰撞重置"""
    #     # 保留原来的超时逻辑
    #     time_out = self.episode_length_buf >= self.max_episode_length - 1
        
    #     # 移除碰撞检测，永远不因为碰撞重置
    #     died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
    #     return died, time_out

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        
        # Check body collision (base contact)
        body_collision = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        
        # Check thigh collision (new addition)
        thigh_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if hasattr(self, '_thigh_ids') and len(self._thigh_ids) > 0:
            thigh_contact_forces = net_contact_forces[:, :, self._thigh_ids]  # [N, T, num_thighs]
            thigh_contact_norm = torch.norm(thigh_contact_forces, dim=-1)  # [N, T, num_thighs]
            thigh_max_contact = torch.max(thigh_contact_norm, dim=1)[0]  # [N, num_thighs]
            thigh_collision = torch.any(thigh_max_contact > 1.0, dim=1)  # [N]
        
        #Combine collision conditions: base OR thigh contact
        collision = body_collision #| thigh_collision
        
        return collision, time_out

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
        threshold_deg = 0
        
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
        # 重置命令计时器和超时时间
        self._command_timer[env_ids] = 0.0
        # 为重置的环境设置新的重新采样间隔时间（1-10秒随机）
        self._command_timeout[env_ids] = torch.zeros_like(self._command_timeout[env_ids]).uniform_(
            self._command_resample_time_min, self._command_resample_time_max
        )
        
        # 只在初始化时采样一次命令，后续由_update_command_resample自动处理
        # X轴: -1 到 1
        # self._commands[env_ids, 0] = torch.zeros_like(self._commands[env_ids, 0]).uniform_(-0, 0)
        # # Y轴: -1 到 1
        # self._commands[env_ids, 1] = torch.zeros_like(self._commands[env_ids, 1]).uniform_(-0, 0)
        # # Z轴(角速度): -1.0 到 1.0
        # self._commands[env_ids, 2] = torch.zeros_like(self._commands[env_ids, 2]).uniform_(-0, 0)
        # # 高度目标: 0.0 到 0.4
        self._commands[env_ids, 3] = torch.zeros_like(self._commands[env_ids, 3]).uniform_(0.15, 0.4)
        
        # Reset robot state - 使用随机化的默认关节位置
        joint_pos = self._randomized_default_joint_pos[env_ids]
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
            zero_prob = 0.3# 30%的概率采样0值
            
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
            self._commands[resample_env_ids, 3] = torch.zeros_like(self._commands[resample_env_ids, 3]).uniform_(0.1, 0.5)
            
            # 重置这些环境的命令计时器
            self._command_timer[resample_env_ids] = 0.0
            
            # 为这些环境设置新的重新采样间隔时间（1-10秒随机）
            self._command_timeout[resample_env_ids] = torch.zeros_like(self._command_timeout[resample_env_ids]).uniform_(
                self._command_resample_time_min, self._command_resample_time_max
            )

    def _compute_feet_stumble_penalty(self) -> torch.Tensor:
        """计算脚部撞击垂直表面的惩罚
        
        当脚部撞击垂直表面时（水平力远大于垂直力），给予惩罚。
        这有助于避免机器人脚部撞击障碍物或墙面。
        
        Returns:
            torch.Tensor: 脚部撞击惩罚值 [N]
        """
        # 获取接触力数据
        if self._contact_sensor.data.net_forces_w is not None:
            # 提取脚部的接触力 [N, 4, 3]
            feet_forces = self._contact_sensor.data.net_forces_w[:, self._feet_ids]  # [N, 4, 3]
            
            # 计算垂直力（z方向）的绝对值
            forces_z = torch.abs(feet_forces[:, :, 2])  # [N, 4]
            
            # 计算水平力（xy平面）的模长
            forces_xy = torch.linalg.norm(feet_forces[:, :, :2], dim=2)  # [N, 4]
            
            # 检测脚部是否撞击垂直表面：水平力 > 4 * 垂直力
            stumble_condition = forces_xy > 4 * forces_z  # [N, 4]
            
            # 检查是否有任何脚部发生撞击
            stumble_penalty = torch.any(stumble_condition, dim=1).float()  # [N]
            
        else:
            # 如果没有接触力数据，则不给惩罚
            stumble_penalty = torch.zeros(self.num_envs, device=self.device)
        
        # 根据机器人姿态调整惩罚强度
        # 使用重力投影的z分量来判断机器人的姿态稳定性
        # projected_gravity_b[:, 2] 范围为 [-1, 1]，-1表示完全正立
        orientation_factor = torch.clamp(-self._robot.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        
        return stumble_penalty * orientation_factor

    