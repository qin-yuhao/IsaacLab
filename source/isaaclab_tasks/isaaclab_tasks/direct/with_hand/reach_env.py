# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG,VisualizationMarkersCfg


@configclass
class ReachEnvCfg(DirectRLEnvCfg):
    # env    
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 27  # 6 arm joints + 21 dexterous hand joints
    observation_space = 57  # 27(dof_pos) + 27(dof_vel) + 3(target_pos) 
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="C:/Users/Admin/Desktop/IsaacLab/robot_with_hand.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1,
                fix_root_link= True,  # Fix the root link to avoid unwanted movements
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": 0.0,
                "joint2": 0.0,
                "joint3": 0.0,
                "joint4": 0.0,
                "joint5": 0.0,
                "joint6": 0.0,
                # Dexterous hand initial positions (slightly curved)
                "thumb_joint.*": 0.0,
                "index_joint.*": 0.0,
                "middle_joint.*": 0.0,
                "ring_joint.*": 0.0,
                "little_joint.*": 0.0,
            },
            pos=(1.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "arm_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-3]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "arm_wrist": ImplicitActuatorCfg(
                joint_names_expr=["joint[4-6]"],
                effort_limit=50.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "dexterous_hand": ImplicitActuatorCfg(
                joint_names_expr=["thumb_joint.*", "index_joint.*", "middle_joint.*", "ring_joint.*", "little_joint.*"],
                effort_limit=20.0,
                velocity_limit=2.6,
                stiffness=100.0,
                damping=5.0,
            ),
        },
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    action_scale = 7.5
    dof_velocity_scale = 0.1    # reward scales for reach task
    reach_reward_scale = 1.0
    action_penalty_scale = 0.01
    dof_acc_penalty_scale = 1e-7  # DOF加速度惩罚权重
    
    # target update settings
    target_update_interval_s = 4.0  # 每2秒更新一次目标位置
    
      # target generation (相对于机器人基座的坐标范围)
    target_pos_range = {
        "x": (-1, 1),  # 前后范围（相对于基座）
        "y": (-1, 1), # 左右范围（相对于基座）
        "z": (0, 1)   # 上下范围（相对于基座）
    }

    visualization_markers: VisualizationMarkersCfg = POSITION_GOAL_MARKER_CFG.replace(
        prim_path="/Visuals/Target")
    


class ReachEnv(DirectRLEnv):
    cfg: ReachEnvCfg

    def __init__(self, cfg: ReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        # Set speed scales for different joint groups
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        
        # Find and set speed scales for dexterous hand joints
        hand_joint_patterns = ["thumb_joint", "index_joint", "middle_joint", "ring_joint", "little_joint"]
        for pattern in hand_joint_patterns:
            hand_joints = self._robot.find_joints(f"{pattern}.*")
            if hand_joints[0]:  # if joints found
                for joint_idx in hand_joints[0]:
                    self.robot_dof_speed_scales[joint_idx] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)        # Target positions for reach task 
        self.target_positions = torch.zeros((self.num_envs, 3), device=self.device)
        #self.hand_link_idx = self._robot.find_bodies("Link6")[0][0]  # 手腕链接
        self.hand_link_idx = self._robot.find_bodies("hand_base")[0][0]  # 手部基座链接        # Target update timer (用于每2秒更新目标位置)
        self.target_update_timer = torch.zeros(self.num_envs, device=self.device)

        # For DOF acceleration penalty
        self.prev_joint_vel = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self._goal_visualizer = VisualizationMarkers(self.cfg.visualization_markers)
        self._goal_visualizer.set_visibility(True)
    def _update_goal_visualization(self):
        """Update the visualization of goal positions by converting relative coordinates to world coordinates."""
        # 获取机器人基座的世界坐标
        robot_base_pos = self._robot.data.root_pos_w
        # 将相对坐标转换为世界坐标进行可视化
        goal_positions_world = self.target_positions + robot_base_pos
        self._goal_visualizer.visualize(translations=goal_positions_world)


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        
        # 更新目标更新计时器
        self.target_update_timer += self.dt
        
        # 检查是否需要更新目标位置（每2秒更新一次）
        update_mask = self.target_update_timer >= self.cfg.target_update_interval_s
        if torch.any(update_mask):
            # 获取需要更新目标的环境ID
            env_ids_to_update = torch.where(update_mask)[0]
            # 更新这些环境的目标位置
            self._generate_random_targets(env_ids_to_update)
            # 重置计时器
            self.target_update_timer[env_ids_to_update] = 0.0
            # 更新可视化
            self._update_goal_visualization()

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 检查是否到达目标 (距离小于5cm视为成功)
        hand_pos = self._robot.data.body_pos_w[:, self.hand_link_idx]
        # 将相对坐标转换为世界坐标进行距离计算
        robot_base_pos = self._robot.data.root_pos_w
        target_positions_world = self.target_positions + robot_base_pos
        distance_to_target = torch.norm(hand_pos - target_positions_world, dim=-1)
        terminated = distance_to_target < 0.05  # 5cm threshold
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # 获取手部位置（世界坐标）
        hand_pos = self._robot.data.body_pos_w[:, self.hand_link_idx]
        
        # 将相对坐标转换为世界坐标进行距离计算
        robot_base_pos = self._robot.data.root_pos_w
        target_positions_world = self.target_positions + robot_base_pos
        distance_to_target = torch.norm(hand_pos - target_positions_world, dim=-1)
        
        # 距离奖励（距离越近奖励越高）
        reach_reward = torch.exp(-distance_to_target/0.25)  # 使用指数衰减函数，距离越近奖励越高
        
        # DOF加速度惩罚
        current_joint_vel = self._robot.data.joint_vel
        joint_acc = (current_joint_vel - self.prev_joint_vel) / self.dt
        dof_acc_penalty = torch.sum(joint_acc**2, dim=-1)
        
        # 更新之前的关节速度
        self.prev_joint_vel[:] = current_joint_vel.clone()
        
        # 动作惩罚
        #action_penalty = torch.sum(self.actions**2, dim=-1)
        
        total_reward = (
            self.cfg.reach_reward_scale * reach_reward
            - self.cfg.dof_acc_penalty_scale * dof_acc_penalty
            #- self.cfg.action_penalty_scale * action_penalty
        )        
        return total_reward

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        
        # 重置机器人关节位置
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
          # 重置目标更新计时器
        self.target_update_timer[env_ids] = 0.0
        
        # 重置之前的关节速度（用于加速度计算）
        self.prev_joint_vel[env_ids] = 0.0
        
          # 生成新的随机目标位置
        self._generate_random_targets(env_ids)
        self._update_goal_visualization()

    def _generate_random_targets(self, env_ids: torch.Tensor):
        """生成相对于机器人基座的随机目标位置"""
        num_resets = len(env_ids)
        
        # 在指定范围内生成相对于机器人基座的随机目标
        target_x = sample_uniform(
            self.cfg.target_pos_range["x"][0], 
            self.cfg.target_pos_range["x"][1], 
            (num_resets, 1), self.device
        )
        target_y = sample_uniform(
            self.cfg.target_pos_range["y"][0], 
            self.cfg.target_pos_range["y"][1], 
            (num_resets, 1), self.device
        )
        target_z = sample_uniform(
            self.cfg.target_pos_range["z"][0], 
            self.cfg.target_pos_range["z"][1], 
            (num_resets, 1), self.device
        )
          # 存储相对于机器人基座的目标位置
        self.target_positions[env_ids] = torch.cat([target_x, target_y, target_z], dim=-1)

    def _get_observations(self) -> dict:
        # 缩放关节位置到[-1, 1]范围
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        
        obs = torch.cat(
            (
                dof_pos_scaled,  # [N, 27] - 所有关节位置
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,  # [N, 27] - 关节速度
                self.target_positions,  # [N, 3] - 相对于机器人基座的目标位置
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}
