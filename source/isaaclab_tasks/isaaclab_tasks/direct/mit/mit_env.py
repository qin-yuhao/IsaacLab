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
from isaaclab.sensors import ContactSensor, RayCaster, Camera,TiledCamera

from .mit_env_cfg import MitFlatEnvCfg, MitRoughEnvCfg
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.utils.math import quat_rotate_inverse

class MitEnv(DirectRLEnv):
    cfg: MitFlatEnvCfg | MitRoughEnvCfg

    def __init__(self, cfg: MitFlatEnvCfg | MitRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        #self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._actions = torch.zeros(self.num_envs, 12, device=self.device)
        # self._previous_actions = torch.zeros(
        #     self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        # ).
        self._previous_actions = torch.zeros(self.num_envs, 12, device=self.device)

        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        # 添加目标速度 (1维向量，范围0.1-1)
        self._target_speed = torch.zeros(self.num_envs, 1, device=self.device)
        # Logging - 添加角速度命令跟踪日志
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_ang_vel_cmd",  # 新增角速度命令跟踪日志
                "track_lin_vel_cmd",  # 新增线速度命令跟踪日志
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "goal_reached",
                "approach_reward",
                "heading_reward",
                "speed_tracking",  # 新增目标速度跟踪奖励
                "upward_reward",  # 新增站立奖励
                "roll_recovery_reward",  # 翻滚恢复奖励
                "foot_velocity_reward",  # 新增足端向下速度奖励
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(["base"])#,".*thigh",".*calf"

        self.set_debug_vis(True)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, MitRoughEnvCfg):
            # we add a height scanner for perceptive locomotion
           # self._height_scanner = RayCaster(self.cfg.height_scanner)
            #self.scene.sensors["height_scanner"] = self._height_scanner

            self._front_lidar = RayCaster(self.cfg.front_lidar)
            self.scene.sensors["front_lidar"] = self._front_lidar

            # self._cam = TiledCamera(self.cfg.cam)
            # self.scene.sensors["cam"] = self._cam
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


        #创建一个光源
        light_cfg2 = sim_utils.SphereLightCfg(
            color=(0.0, 0.75, 0.75),
            intensity=2000.0,
            radius=0.5,
        )
        light_cfg2.func("/World/SphereLight", light_cfg2)
        # 保存需要跟踪的目标路径
        self._light_path = "/World/SphereLight"
        
        # 确认该路径存在
        if prim_utils.is_prim_path_valid(self._light_path):
            print(f"找到了目标光源: {self._light_path}")

    def _get_light_position(self):
        """获取灯光的实时位置"""
        
        # 获取变换属性
        light_prim = prim_utils.get_prim_at_path(self._light_path)
        
        # 使用xformOp:translate属性获取位置
        if light_prim.HasAttribute("xformOp:translate"):
            pos = light_prim.GetAttribute("xformOp:translate").Get()
            return torch.tensor(pos, device=self.device)
            
    


    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        # 使用命令作为目标点位置
        # 获取机器人位置和旋转
        robot_pos = self._robot.data.root_pos_w
        robot_rot = self._robot.data.root_quat_w
        
        # 计算目标点与机器人的相对位置（世界坐标系）
        rel_pos_w = self._commands - robot_pos
        
        # 计算水平距离（忽略高度）
        horizontal_distance = torch.norm(rel_pos_w[:, :2], dim=1).unsqueeze(1)  # 只考虑x-y平面
        #horizontal_distance_normalized = torch.clamp(horizontal_distance / 2, 0.0, 1.0)  # 归一化到[0,1]
        
        # 计算相对偏航角
        # 创建单位向量 [1,0,0] 表示x轴方向（前向）
        forward_dir = torch.zeros((self.num_envs, 3), device=self.device)
        forward_dir[:, 0] = 1.0  # x轴方向
        
        # 使用机器人的四元数旋转该向量
        forward_vec_w = quat_rotate_inverse(robot_rot, forward_dir)
        
        # 只取x-y平面的分量用于计算偏航角
        forward_vec = forward_vec_w[:, :2]
        forward_vec = torch.nn.functional.normalize(forward_vec, dim=1)
        
        # 计算目标方向向量（水平面上）
        target_dir_w = torch.nn.functional.normalize(rel_pos_w[:, :2], dim=1)
        
        # 计算向量之间的夹角（点积）
        cos_angle = torch.sum(forward_vec * target_dir_w, dim=1).unsqueeze(1)
        # 交叉积的z分量，用于判断左右
        cross_z = (forward_vec[:, 0] * target_dir_w[:, 1] - forward_vec[:, 1] * target_dir_w[:, 0]).unsqueeze(1)
        sin_angle = cross_z
        
        # 将夹角信息作为观测
        rel_yaw = torch.cat([cos_angle, sin_angle], dim=1)  # [cos(θ), sin(θ)]表示相对偏
        # 原有代码部分
        # height_data = None
        # if isinstance(self.cfg, MitRoughEnvCfg):
        #     height_data = (
        #         self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
        #     ).clip(-1.0, 1.0)

        lidar_data = None
        if isinstance(self.cfg, MitRoughEnvCfg):
            lidar_data = self._front_lidar.data.ray_hits_w[:, :, 2] - self._front_lidar.data.pos_w[:, 2].unsqueeze(1)
            lidar_data = lidar_data.clip(-5.0, 5.0)
            lidar_data = lidar_data.view(self.num_envs, -1)
        
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,               # 3: 速度
                    self._robot.data.root_ang_vel_b,               # 3: 角速度
                    self._robot.data.projected_gravity_b,          # 3: 重力方向
                    rel_yaw,                                       # 2: 目标相对偏航角[cos, sin]
                    horizontal_distance,                # 1: 到目标的水平距离
                    self._target_speed,                               # 1: 目标速度
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos, # 12: 关节位置
                    self._robot.data.joint_vel,                    # 12: 关节速度
                    #height_data,                                   # 187: 高度数据
                    lidar_data,                                   # 64: 激光雷达数据
                    self._actions,                                 # 12: 动作
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:

        
        #goal tracking
        #light_position = self._get_light_position()
        robot_pos = self._robot.data.root_pos_w
        robot_rot = self._robot.data.root_quat_w
        
        # 计算机器人朝向（通过重力投影判断姿态）
        projected_gravity_z = self._robot.data.projected_gravity_b[:, 2]
        
        # 判断是否处于倒地状态
        # projected_gravity_z > 0 表示机器人倒立或侧翻
        is_flipped = projected_gravity_z > 0.0
        
        # 获取足端位置
        # 假设足端名称包含"_foot"，您可能需要根据实际模型调整
        # 获取足端位置和速度
        foot_ids = []
        for i, body_name in enumerate(self._robot.data.body_names):
            if "_calf" in body_name.lower():  # 找到所有含"calf"的身体部件（小腿末端）
                if "L_calf" in body_name or "FL_calf" in body_name or "RL_calf" in body_name:
                    foot_ids.append(i)

        # 当机器人倒地时，不再奖励低足端高度，而是奖励足端向下的速度
        foot_velocities = []
        for foot_id in foot_ids:
            # 获取足端在世界坐标系中的速度
            foot_vel_w = self._robot.data.body_vel_w[:, foot_id]
            # 只关注竖直方向的速度（z轴），向下为负值
            foot_vel_z = -foot_vel_w[:, 2]  # 取负值，使向下为正
            foot_velocities.append(foot_vel_z)

        # 将所有足端速度堆叠为单个张量 [num_envs, num_feet]
        foot_velocities_tensor = torch.stack(foot_velocities, dim=1)

        # 计算平均向下速度（已转换为正值）
        mean_foot_downward_velocity = torch.mean(foot_velocities_tensor, dim=1)

        # 当倒地时，奖励足端向下的速度
        # 速度越大，奖励越高，但设置上限避免过度激励
        foot_velocity_reward = torch.where(
            is_flipped,
            torch.clamp(mean_foot_downward_velocity, 0.0, 3.0) / 3.0,  # 归一化到[0,1]
            torch.zeros_like(mean_foot_downward_velocity)  # 正常站立时没有此奖励
        )

        # 添加一个非线性函数，使得速度越大奖励增长越快
        foot_velocity_reward = torch.pow(foot_velocity_reward, 2)  # 平方函数使速度越大奖励越高

        
        # -1代表正确站立，接近1代表倒立
        # 转化为0到1的奖励，1代表完全站立，0代表完全倒立
        upward_reward = (-projected_gravity_z + 1) / 2
        
        # 添加非线性奖励曲线，使得接近站立时奖励更高
        upward_reward = torch.pow(upward_reward, 2)  # 平方函数使接近1时奖励更高
        
        # 添加翻滚角速度奖励 - 仅在仰面朝上时激活
        upside_down_mask = projected_gravity_z > 0.8 # 如果z轴重力投影>0.5，认为是仰面朝上
        roll_angular_velocity = self._robot.data.root_ang_vel_b[:, 0]  # x轴角速度（翻滚）
        
        # 创建奖励：
        # 1. 如果机器人仰面朝上，奖励适当的翻转角速度（正向或负向都可以，取绝对值）
        # 2. 如果不是仰面朝上，则该奖励为0
        roll_recovery_reward = torch.zeros_like(roll_angular_velocity)
        
        # 当仰面朝上时，任何翻转动作都会得到奖励（绝对值）
        roll_recovery_reward = roll_angular_velocity
          
        
        # 调整奖励大小，使其在合适的范围内（例如0-1）
        #roll_recovery_reward = torch.clamp(roll_recovery_reward / 5.0, 0.0, 1.0)
        #roll_recovery_reward = torch.pow(roll_recovery_reward, 2)  # 平方函数使接近1时奖励更高

        robot_pos = self._robot.data.root_pos_w
        robot_rot = self._robot.data.root_quat_w
        robot_vel = self._robot.data.root_lin_vel_w
    
        # 计算到目标点的距离
        rel_pos_w = self._commands - robot_pos
        distance_to_goal = torch.norm(rel_pos_w[:, :2], dim=1)  # 只考虑水平距离
        
        progress_buffer = self.episode_length_buf.float() #/ self.max_episode_length
        
        # 检查是否本次步骤机器人达到了done状态（检测上一次步骤是否为done，本次是否计算过奖励）
        # 注：正常情况下done后环境会被重置，这里是为了确保在重置前给予奖励
        done_this_step = self.reset_terminated | self.reset_time_outs
        
        # 计算机器人的前向向量（在世界坐标系中）
        forward_dir = torch.zeros((self.num_envs, 3), device=self.device)
        forward_dir[:, 0] = 1.0  # x轴方向
        forward_vec_w = quat_rotate_inverse(robot_rot, forward_dir)
        forward_vec_w_xy = forward_vec_w[:, :2]  # 只取水平面分量
        forward_vec_w_xy = torch.nn.functional.normalize(forward_vec_w_xy, dim=1)
        
        # 计算目标方向向量（水平面）
        target_dir_w = torch.nn.functional.normalize(rel_pos_w[:, :2], dim=1)
        
        # 计算朝向对齐度（点积）- 范围从[-1,1]，1表示完全对齐
        heading_alignment = torch.sum(forward_vec_w_xy * target_dir_w, dim=1)
        
        # 只有当距离小于0.2米时才给予终止目标奖励
        termination_goal_reward = torch.where(
            (done_this_step) & (distance_to_goal < 0.2),  # 增加距离条件
            torch.ones_like(distance_to_goal) * self.cfg.goal_reward_scale * progress_buffer * self.step_dt,
            torch.zeros_like(distance_to_goal)
        )
        
        termination_approach_reward = torch.where(
            done_this_step,
            torch.exp(-distance_to_goal / 1) * self.cfg.approach_reward_scale * progress_buffer * self.step_dt,
            torch.zeros_like(distance_to_goal)
        )

        target_speed = self._target_speed.squeeze(1)  # [num_envs]
        forward_speed = robot_vel[:, 0]  # 局部坐标系中的X轴速度
        #  #不能低于0
        speed_error = target_speed - forward_speed
        speed_error = torch.clamp(speed_error, min=0.0)  # 不能低于0
       
        speed_reward = torch.exp(-speed_error / 0.25)  # 指数衰减，误差为0时奖励为1，误差越大奖励越小
        

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

    
        rewards = {
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "goal_reached":  termination_goal_reward ,
            "approach_reward": termination_approach_reward ,
             "heading_reward": heading_alignment * self.cfg.heading_reward_scale * self.step_dt,
             "speed_tracking": speed_reward * self.cfg.speed_tracking_reward_scale * self.step_dt,
             "upward_reward": upward_reward * self.cfg.upward_reward_scale * self.step_dt,
             "roll_recovery_reward": roll_recovery_reward * self.cfg.roll_recovery_scale * self.step_dt,
             "foot_velocity_reward": foot_velocity_reward * self.cfg.foot_height_reward_scale * self.step_dt,  # 使用相同的缩放因子
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value


        # clip to no less than 0
        #reward = torch.clamp(reward, min=0.0)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 检查时间是否结束
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # 检查基座碰撞
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        base_collision = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        
        # 检查机器狗高度是否低于-2米
        robot_height = self._robot.data.root_pos_w[:, 2]
        height_too_low = robot_height < -4
        
        # 检查是否接近目标点 (距离 < 0.1米)
        rel_pos_w = self._commands - self._robot.data.root_pos_w
        distance_to_goal = torch.norm(rel_pos_w[:, :2], dim=1)  # 只考虑水平距离
        goal_reached = distance_to_goal < 0.1
        
        # 终止条件：基座碰撞、高度过低或接近目标点
        died = base_collision | height_too_low
        #died = height_too_low
        
        # 目标达成作为一种特殊的终止条件（成功终止）
        return died, time_out | goal_reached

    # def _reset_idx(self, env_ids: torch.Tensor | None):
    #     if env_ids is None or len(env_ids) == self.num_envs:
    #         env_ids = self._robot._ALL_INDICES
    #     self._robot.reset(env_ids)
    #     super()._reset_idx(env_ids)
    #     if len(env_ids) == self.num_envs:
    #         # Spread out the resets to avoid spikes in training when many environments reset at a similar time
    #         self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
    #     self._actions[env_ids] = 0.0
    #     self._previous_actions[env_ids] = 0.0
        
    #     # 获取机器人的初始位置和朝向
    #     joint_pos = self._robot.data.default_joint_pos[env_ids]
    #     joint_vel = self._robot.data.default_joint_vel[env_ids]
    #     default_root_state = self._robot.data.default_root_state[env_ids].clone()
    #     default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
    #     # 设置目标点在机器人前方2米处（基于世界坐标系）
    #     # 获取机器人前向量（假设前方是x轴正方向）
    #     robot_positions = default_root_state[:, :3]
        
    #     # 设置命令为世界坐标系中的目标点位置 加上随机偏移
    #     self._commands[env_ids, 0] = robot_positions[:, 0] + torch.zeros_like(robot_positions[:, 0], device=self.device).uniform_(2, 5) 
    #     self._commands[env_ids, 1] = robot_positions[:, 1] + torch.zeros_like(robot_positions[:, 1], device=self.device).uniform_(-3, 3)
    #     self._commands[env_ids, 2] = robot_positions[:, 2]   
        
    #     # 设置随机目标速度 (0.1-1.0 m/s)
    #     self._target_speed[env_ids] = 0.3 + 0.9 * torch.rand(len(env_ids), 1, device=self.device)

    #     # 写入机器人状态
    #     self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
    #     self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
    #     self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    
    #     # Logging
    #     extras = dict()
    #     for key in self._episode_sums.keys():
    #         episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
    #         extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
    #         self._episode_sums[key][env_ids] = 0.0
    #     self.extras["log"] = dict()
    #     self.extras["log"].update(extras)
    #     extras = dict()
    #     extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
    #     extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
    #     self.extras["log"].update(extras)

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
        robot_positions = self._robot.data.root_pos_w[env_ids]
        self._commands[env_ids, 0] = robot_positions[:, 0] + torch.zeros_like(robot_positions[:, 0], device=self.device).uniform_(2, 5) 
        self._commands[env_ids, 1] = robot_positions[:, 1] + torch.zeros_like(robot_positions[:, 1], device=self.device).uniform_(-3, 3)
        self._commands[env_ids, 2] = robot_positions[:, 2]   
        
        # 设置随机目标速度
        self._target_speed[env_ids] = 0.3 + 0.9 * torch.rand(len(env_ids), 1, device=self.device)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._robot.write_data_to_sim()
        
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
    def _set_debug_vis_impl(self, debug_vis):
        if not hasattr(self, "_goal_marker"):
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG
            
            # 创建球体标记配置，设置为绿色
            marker_cfg = POSITION_GOAL_MARKER_CFG.replace(prim_path="/Visuals/GoalMarkers")
            marker_cfg.markers["target_far"].radius = 0.1
            self._goal_marker = VisualizationMarkers(
                marker_cfg
            )

    def _debug_vis_callback(self, event):
        """更新目标点的可视化"""
        if hasattr(self, "_goal_marker"):
            # 更新目标点位置
            goal_positions = self._commands.cpu().numpy()
        
            # 更新目标点位置
            self._goal_marker.visualize(translations=goal_positions)
            
            # 确保可见性
            #self._goal_marker.set_visibility(True)
