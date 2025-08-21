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
from isaaclab.sensors import ContactSensor, RayCaster

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
        self._mode = torch.zeros(self.num_envs, 1, device=self.device)

        # 添加命令计时器和切换设置
        self._command_change_interval = 10 # 每5秒改变一次命令和模式
        self._command_change_time = torch.zeros(self.num_envs, device=self.device)  # 跟踪每个环境的下一次改变时间
        self._command_prob_stay = 0.7  # 保持当前模式的概率
        self._command_timer = torch.zeros(self.num_envs, device=self.device)  # 跟踪每个环境的命令时间
        
        self._fallen_timer = torch.zeros(self.num_envs, device=self.device)  # 跟踪每个环境倒地的持续时间
        self._fallen_timeout =2.5  # 倒地超过3秒触发重置
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
                "joint_vel_l2",  # 添加关节速度日志
                "upward_reward",  # 添加站立奖励日志
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*_foot")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*_thigh")

        self._init_pose_variants = {
            "standing": {
            ".*L_hip_joint": 0.1, #弧度转角度： 0.1*180/3.14 = 5.73
            ".*R_hip_joint": -0.1, #5.73
            "F[L,R]_thigh_joint":-0.6, #-0.6*180/3.14 = -34.38
            "R[L,R]_thigh_joint": 0.6, #34.38
            "F[L,R]_calf_joint":1.4, #1.4*180/3.14 = 80.10
            "R[L,R]_calf_joint": -1.4, #-80.10
        },
            "lie_down": {
                "F[L,R]_hip_joint": -45 * torch.pi / 180.0,
                "R [L,R]_hip_joint": -35 * torch.pi / 180.0,
                "F[L,R]_thigh_joint": -55 * torch.pi / 180.0,
                "R[L,R]_thigh_joint": 63 * torch.pi / 180.0,
                "F[L,R]_calf_joint": 145 * torch.pi / 180.0,
                "R[L,R]_calf_joint": -155 * torch.pi / 180.0,
        },
            "walking": {
                "F[L,R]_hip_joint": 0.0,
                "R[L,R]_hip_joint": -0.0,
                "FL_thigh_joint": -11 * torch.pi / 180.0,
                "FL_calf_joint": 94 * torch.pi / 180.0,
                "FR_thigh_joint": -52 * torch.pi / 180.0,
                "FR_calf_joint": 94 * torch.pi / 180.0,
                "RL_thigh_joint": 46* torch.pi / 180.0,
                "RL_calf_joint": -82 * torch.pi / 180.0,
                "RR_thigh_joint": 14 * torch.pi / 180.0,
                "RR_calf_joint": -82 * torch.pi / 180.0,
        },
        }




    def _sample_commands_and_modes(self, env_ids: torch.Tensor):
        """为指定的环境采样新的命令
        
        Args:
            env_ids: 要更新的环境ID
        """
        # 采样新的速度命令
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        # 删除所有模式相关代码

    def step(self, actions: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # 更新命令计时器
        self._command_timer += self.step_dt
        
        # 检查哪些环境需要更新命令
        change_command_mask = self._command_timer >= self._command_change_time
        env_ids_change = torch.nonzero(change_command_mask).squeeze(-1)
        
        if len(env_ids_change) > 0:
            # 为需要更新的环境设置新的命令和模式
            self._sample_commands_and_modes(env_ids_change)
            
            # 更新这些环境的下一次改变时间
            self._command_change_time[env_ids_change] = self._command_timer[env_ids_change] + self._command_change_interval
        
        # 调用原来的step逻辑
        return super().step(actions)
    

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
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
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None
        if isinstance(self.cfg, MitRoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    height_data,
                    self._actions,
                    # 删除 self._mode
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _compute_velocity_tracking_reward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """计算线速度和偏航速率跟踪奖励
        Returns:
            tuple: (线速度跟踪奖励映射值, 偏航速率跟踪奖励映射值)
        """
        # 获取站立状态 - 判断机器人是否站立良好
        projected_gravity_z = self._robot.data.projected_gravity_b[:, 2]
        safe_values = torch.clamp(projected_gravity_z, -1.0, 1.0) 
        angle_rad = torch.acos(-safe_values)
        angle_deg = angle_rad * 180.0 / torch.pi  # 转为角度
        
        # 站立良好的阈值 - 与upward_reward中使用相同的阈值
        max_stand_angle = 60.0
        standing_well_mask = angle_deg <= max_stand_angle
        # 计算线速度和偏航速率的误差
        # 仅在站立良好的环境中计算奖励
        # 线速度误差
        lin_vel_error = torch.sum(torch.square(self._robot.data.root_lin_vel_b[:, :2] - self._commands[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # 偏航速率误差
        yaw_rate_error = torch.square(self._robot.data.root_ang_vel_b[:, 2] - self._commands[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25) 
        
        # 计算奖励映射值
        # 线速度跟踪奖励映射值
        lin_vel_error_mapped = torch.where(
            standing_well_mask,
            lin_vel_error_mapped,
            torch.zeros_like(lin_vel_error_mapped)  # 站立不良时，奖励为0
        )
        # 偏航速率跟踪奖励映射值
        yaw_rate_error_mapped = torch.where(
            standing_well_mask,
            yaw_rate_error_mapped,
            torch.zeros_like(yaw_rate_error_mapped)  # 站立不良时，奖励为0
        )
        return lin_vel_error_mapped, yaw_rate_error_mapped


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
        threshold_deg = 25.0
        
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

    def _get_rewards(self) -> torch.Tensor:
       # 使用新函数计算速度跟踪奖励
        lin_vel_error_mapped, yaw_rate_error_mapped = self._compute_velocity_tracking_reward()
    
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
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
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

        # 添加关节速度惩罚 - 惩罚过大的关节速度
        joint_vel_penalty = torch.sum(torch.square(self._robot.data.joint_vel), dim=1)

        upward_reward = self._compute_upward_reward()
        
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
            "joint_vel_l2": joint_vel_penalty * self.cfg.joint_vel_reward_scale * self.step_dt,  # 新增关节速度惩罚
            "upward_reward": upward_reward * self.cfg.upward_reward_scale * self.step_dt,  # 新增站立奖励
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 普通的超时检测
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # 检测机器人的姿态
        projected_gravity_z = self._robot.data.projected_gravity_b[:, 2]
        safe_values = torch.clamp(projected_gravity_z, -1.0, 1.0)
        angle_rad = torch.acos(-safe_values)
        angle_deg = angle_rad * 180.0 / torch.pi
        
        # 判断是否倒地
        fallen_threshold = 90.0  # 超过90度认为倒地
        is_fallen = angle_deg > fallen_threshold
        
        # 更新倒地计时器
        # 如果倒地，则累加时间；否则重置计时器
        self._fallen_timer = torch.where(
            is_fallen,
            self._fallen_timer + self.step_dt,  # 累加时间
            torch.zeros_like(self._fallen_timer)  # 重置计时器
        )
        
        # 检查是否倒地超时
        fallen_timeout = self._fallen_timer >= self._fallen_timeout
        
        # 机器人高度过低也视为终止
        # robot_height = self._robot.data.root_pos_w[:, 2]
        # height_too_low = robot_height < -4
        
        # 组合终止条件：倒地超时或高度过低
        died = fallen_timeout #| height_too_low
        
        return died, time_out

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
        
        # 重置命令相关定时器
        self._command_timer[env_ids] = 0.0
        self._command_change_time[env_ids] = (torch.rand(len(env_ids), device=self.device)+0.3) * self._command_change_interval
        self._fallen_timer[env_ids] = 0.0
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        #self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        #self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
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
