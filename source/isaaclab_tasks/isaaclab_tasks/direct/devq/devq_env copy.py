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

from .devq_env_cfg import DevqFlatEnvCfg, DevqRoughEnvCfg


class DevqEnv(DirectRLEnv):
    cfg: DevqFlatEnvCfg | DevqRoughEnvCfg

    def __init__(self, cfg: DevqFlatEnvCfg | DevqRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        self._prev_contact_state = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)
        self._prev_feet_pos = torch.zeros((self.num_envs, 4, 3), device=self.device)
        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        self._initial_root_pos_x = torch.zeros(self.num_envs, device=self.device)
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
                "undesired_contacts",
                "flat_orientation_l2",
                "hip_deviation_penalty", 
                "standing_pos_penalty", 
                "feet_air_time",
            ]

        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("body")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        #self._feet_ids= [13, 14, 15, 16]
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(["front_left_abad_link", "front_left_hip_link",  "front_right_abad_link", "front_right_hip_link", "hind_left_abad_link", "hind_left_hip_link",  "hind_right_abad_link", "hind_right_hip_link", "front_right_knee_link" , "hind_right_knee_link","body", "hind_left_knee_link","front_left_knee_link",]) #

        # 添加速度历史缓冲区
        self.velocity_window_size = 5  # 使用10步来计算平均速度
        self._lin_vel_buffer = torch.zeros((self.num_envs, self.velocity_window_size, 2), device=self.device)
        self._ang_vel_buffer = torch.zeros((self.num_envs, self.velocity_window_size, 1), device=self.device)
        self._buffer_index = 0 

        # 添加观测历史缓冲区 - 存储10帧历史观测
        self.obs_history_length = 10  # 存储10帧历史
        self._obs_history_buffer = None  # 将在第一次获取观测时初始化
        self._obs_history_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 每个环境的历史索引

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # Add IMU sensor
        self._imu_sensor = Imu(self.cfg.imu)
        self.scene.sensors["imu_sensor"] = self._imu_sensor
        if isinstance(self.cfg, DevqRoughEnvCfg):
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
        #增大小腿关节的幅度，后4个关节，8，9，10，11乘以2
        #self._actions[:, 8:12] *= 2.0

        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

        # 更新速度缓存
        self._lin_vel_buffer[:, self._buffer_index] = self._robot.data.root_lin_vel_b[:, :2]
        self._ang_vel_buffer[:, self._buffer_index, 0] = self._robot.data.root_ang_vel_b[:, 2]
        self._buffer_index = (self._buffer_index + 1) % self.velocity_window_size

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
        ang_vel_noise = torch.empty_like(imu_ang_vel).uniform_(-0.1, 0.1)
        imu_ang_vel = imu_ang_vel + ang_vel_noise  

        # IMU Orientation (with noise ±0.05)
        imu_lin_acc = self._imu_sensor.data.lin_acc_b
        orientation_noise = torch.empty_like(imu_lin_acc).uniform_(-0.03, 0.03)
        imu_lin_acc = imu_lin_acc + orientation_noise

        # Projected Gravity (with noise ±0.05)
        proj_gravity = self._robot.data.projected_gravity_b  # [3]
        gravity_noise = torch.empty_like(proj_gravity).uniform_(-0.05, 0.05)
        proj_gravity = proj_gravity + gravity_noise

        # Joint Positions (with noise ±0.01)
        joint_positions = self._robot.data.joint_pos - self._robot.data.default_joint_pos  # [12]
        joint_pos_noise = torch.empty_like(joint_positions).uniform_(-0.01, 0.01)
        joint_positions = joint_positions + joint_pos_noise

        # Joint Velocities (with noise ±1.5)
        joint_velocities = self._robot.data.joint_vel  # [12]
        joint_vel_noise = torch.empty_like(joint_velocities).uniform_(-1.5, 1.5)
        joint_velocities = joint_velocities + joint_vel_noise

        # Last Actions (no noise)
        actions = self._actions  # [12]

        height_data = None
        if isinstance(self.cfg, DevqRoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
            
        # 将当前帧的观测组合起来
        current_obs = torch.cat(
            [
                tensor
                for tensor in (
                velocity_commands,    # [3]
                imu_ang_vel,         # [3]
                imu_lin_acc,     # [3]
                proj_gravity,        # [3]
                joint_positions,     # [12]
                joint_velocities,    # [12]
                actions,            # [12]
                height_data,    #可选
                )
                if tensor is not None
            ],
            dim=-1,
        )
        
        # 计算单个观测的维度
        single_obs_dim = current_obs.shape[1]
        
        # 初始化历史观测缓冲区(如果尚未初始化)
        if self._obs_history_buffer is None:
            self._obs_history_buffer = torch.zeros(
                (self.num_envs, self.obs_history_length, single_obs_dim), 
                device=self.device
            )
        
        # 更新每个环境的观测历史缓冲区
        for env_idx in range(self.num_envs):
            idx = self._obs_history_indices[env_idx]
            self._obs_history_buffer[env_idx, idx] = current_obs[env_idx]
            self._obs_history_indices[env_idx] = (idx + 1) % self.obs_history_length
        
        # 为每个环境重新排序历史缓冲区，确保最新的观测在最后
        flattened_history = []
        for env_idx in range(self.num_envs):
            current_idx = self._obs_history_indices[env_idx]
            # 生成排序的索引，确保最新的观测在最后
            ordered_indices = [(current_idx + i) % self.obs_history_length for i in range(self.obs_history_length)]
            env_history = self._obs_history_buffer[env_idx, ordered_indices].reshape(1, -1)
            flattened_history.append(env_history)
        
        # 将所有环境的历史数据堆叠成一个张量
        flattened_history = torch.cat(flattened_history, dim=0)
        
        observations = {"policy": flattened_history}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        avg_lin_vel = torch.mean(self._lin_vel_buffer, dim=1)  # [num_envs, 2]
        avg_ang_vel = torch.mean(self._ang_vel_buffer, dim=1).squeeze(-1)  # [num_envs]
        
        # 使用平均速度计算奖励
        # 检查命令速度是否小于0.2
        command_magnitude = torch.norm(self._commands[:, :2], dim=1)
        small_commands = command_magnitude < 0.2
        
        # 对于小命令，给予最大奖励1.0；对于其他，使用正常的指数奖励计算
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - avg_lin_vel), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        
        # 对于小命令的环境，给予最大奖励
        lin_vel_error_mapped = torch.where(small_commands, 
                                          torch.ones_like(lin_vel_error_mapped), 
                                          lin_vel_error_mapped)
        
        yaw_rate_error = torch.square(self._commands[:, 2] - avg_ang_vel)
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
        
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1) 
        
        
        # flat orientation 
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        
        # Hip joint deviation penalty (first 4 joints)
        hip_positions = self._robot.data.joint_pos[:, :4]  # Get first 4 joints (hip joints)
        hip_defaults = self._robot.data.default_joint_pos[:, :4]
        hip_deviation = torch.sum(torch.square(hip_positions - hip_defaults), dim=1)
        hip_deviation_penalty = hip_deviation * (torch.norm(self._commands[:, :2], dim=1) > 0.05)

        # Standing joint position deviation penalty
        standing = (torch.norm(self._commands[:, :2], dim=1) < 0.2)  # True when not moving
        joint_positions = self._robot.data.joint_pos
        joint_defaults = self._robot.data.default_joint_pos
        standing_pos_deviation = torch.sum(torch.square(joint_positions - joint_defaults), dim=1)
        standing_pos_penalty = standing_pos_deviation * standing
        
         # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.25) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )

 
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "hip_deviation_penalty": hip_deviation_penalty * self.cfg.hip_deviation_penalty_scale * self.step_dt,
            "standing_pos_penalty": standing_pos_penalty * self.cfg.standing_pos_penalty_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        #reward = torch.clamp(reward, min=0.0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        # Check body collision
        body_collision = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        #永远不重置
        #body_collision  = torch.zeros_like(body_collision)
        # Check body height
        # body_height = self._robot.data.root_pos_w[:, 2]  # Get body z-position
        # body_too_low = body_height < 0.0  # Reset when body goes below ground
        # died = body_collision | body_too_low
        #died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return body_collision, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._prev_contact_state[env_ids] = False 
       
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        
       # Sample new commands with positive X velocity
        new_commands = torch.zeros_like(self._commands[env_ids])
        # x方向只采样正值 [0.0, 1.0]
        new_commands[:, 0] = torch.rand_like(new_commands[:, 0])* 1.4-0.7 # x velocity (0 to 1)
        # y方向依然在 [-1.0, 1.0] 范围内
        new_commands[:, 1] = torch.rand_like(new_commands[:, 1]) * 1.4-0.7# y velocity (-1 to 1)
        # z方向依然在 [-1.0, 1.0] 范围内
        new_commands[:, 2] = torch.rand_like(new_commands[:, 2]) * 1.4-0.7  # yaw velocity (-1 to 1)
        
        self._commands[env_ids] = new_commands
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._initial_root_pos_x[env_ids] = self._robot.data.root_pos_w[env_ids, 0]

        if self._obs_history_buffer is not None:
            self._obs_history_buffer[env_ids] = 0.0
            self._obs_history_indices[env_ids] = 0  # 重置历史索引
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
        extras["Episode_Termination/body_height"] = torch.count_nonzero(self._robot.data.root_pos_w[env_ids, 2] < 0.0).item()
        self.extras["log"].update(extras)
