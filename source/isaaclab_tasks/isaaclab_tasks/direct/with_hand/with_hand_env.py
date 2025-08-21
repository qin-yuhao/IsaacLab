# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform


@configclass
class WithHandEnvCfg(DirectRLEnvCfg):
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
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1,
                fix_root_link=True,  # Fix the root link to prevent it from moving
            ),
        ),        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": 0,
                "joint2": 0,
                "joint3": 0,
                "joint4": 0,
                "joint5": 0,
                "joint6": 0,
                # Dexterous hand initial positions (slightly curved)
                "thumb_joint.*": 0.0,
                "index_joint.*": 0.0,
                "middle_joint.*": 0.0,
                "ring_joint.*": 0.0,
                "little_joint.*": 0.0,
            },
            pos=(1.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),        actuators={
            "arm_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-3]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "arm_wrist": ImplicitActuatorCfg(
                joint_names_expr=["joint[4-6]"],
                effort_limit=80.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "dexterous_hand": ImplicitActuatorCfg(
                joint_names_expr=["thumb_joint.*", "index_joint.*", "middle_joint.*", "ring_joint.*", "little_joint.*"],
                effort_limit=20.0,
                velocity_limit=2.0,
                stiffness=100.0,
                damping=5.0,
            ),        },
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
            restitution=0.0,        ),
    )

    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales for reach task
    reach_reward_scale = 2.0
    action_penalty_scale = 0.01
    
    # target generation
    target_pos_range = {
        "x": (0.2, 0.8),  # 前后范围
        "y": (-0.4, 0.4), # 左右范围
        "z": (0.2, 0.8)   # 上下范围
    }


class WithHandEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: WithHandEnvCfg

    def __init__(self, cfg: WithHandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

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

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        # Target positions for reach task 
        self.target_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_link_idx = self._robot.find_bodies("Link6")[0][0]  # 手腕链接
        
        # 获取灵巧手各手指的pose
        index_finger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/index_link3")),
            self.device,
        )

        thumb_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/thumb_link4")),
            self.device,
        )

        little_finger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/little_link3")),
            self.device,
        )

        middle_finger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/middle_link3")),
            self.device,
        )

        ring_finger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/ring_link3")),
            self.device,
        )

        # 计算手掌中心位置（使用所有手指tip的平均位置）
        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (index_finger_pose[0:3] + thumb_pose[0:3] + middle_finger_pose[0:3] + 
                           ring_finger_pose[0:3] + little_finger_pose[0:3]) / 5.0
        finger_pose[3:7] = index_finger_pose[3:7]  # Use index finger orientation as reference
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )        # 获取手部和手指链接索引
        self.hand_link_idx = self._robot.find_bodies("Link6")[0][0]  # 手腕链接（机械臂末端）
        
        # 获取灵巧手各手指tip链接索引
        try:
            self.thumb_tip_idx = self._robot.find_bodies("thumb_link4")[0][0]
            self.index_tip_idx = self._robot.find_bodies("index_link3")[0][0] 
            self.middle_tip_idx = self._robot.find_bodies("middle_link3")[0][0]
            self.ring_tip_idx = self._robot.find_bodies("ring_link3")[0][0]
            self.little_tip_idx = self._robot.find_bodies("little_link3")[0][0]
        except:
            # 如果找不到链接，使用默认值或手腕链接
            print("Warning: Could not find some finger tip links, using hand link as fallback")
            self.thumb_tip_idx = self.hand_link_idx
            self.index_tip_idx = self.hand_link_idx
            self.middle_tip_idx = self.hand_link_idx
            self.ring_tip_idx = self.hand_link_idx
            self.little_tip_idx = self.hand_link_idx
            
        # 用于兼容性的左右手指索引（可以指向食指和拇指）
        self.left_finger_link_idx = self.index_tip_idx
        self.right_finger_link_idx = self.thumb_tip_idx
        
        self.drawer_link_idx = self._cabinet.find_bodies("drawer_top")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.drawer_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.drawer_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

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
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self._cabinet.data.joint_pos[:, 3] > 0.39
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

        return self._compute_rewards(
            self.actions,
            self._cabinet.data.joint_pos,
            self.robot_grasp_pos,
            self.drawer_grasp_pos,
            self.robot_grasp_rot,
            self.drawer_grasp_rot,
            robot_left_finger_pos,
            robot_right_finger_pos,
            self.gripper_forward_axis,
            self.drawer_inward_axis,
            self.gripper_up_axis,
            self.drawer_up_axis,
            self.num_envs,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.open_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.finger_reward_scale,
            self._robot.data.joint_pos,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # cabinet state
        zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        # 缩放关节位置到[-1, 1]范围
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        
        # 到目标的向量
        to_target = self.drawer_grasp_pos - self.robot_grasp_pos
        
        # 获取各手指tip的位置信息（相对于手腕）
        hand_pos = self._robot.data.body_pos_w[:, self.hand_link_idx]
        finger_positions = []
        try:
            # 收集所有手指tip的相对位置
            finger_tips = [self.thumb_tip_idx, self.index_tip_idx, self.middle_tip_idx, 
                          self.ring_tip_idx, self.little_tip_idx]
            for tip_idx in finger_tips:
                tip_pos = self._robot.data.body_pos_w[:, tip_idx]
                relative_pos = tip_pos - hand_pos  # 相对于手腕的位置
                finger_positions.append(relative_pos)
            finger_positions = torch.cat(finger_positions, dim=-1)  # [N, 15] (3*5 fingers)
        except:
            # 如果无法获取手指位置，使用零向量
            finger_positions = torch.zeros((self.num_envs, 15), device=self.device)

        obs = torch.cat(
            (
                dof_pos_scaled,  # [N, 27] - 所有关节位置
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,  # [N, 27] - 关节速度
                to_target,  # [N, 3] - 到目标的向量
                self._cabinet.data.joint_pos[:, 3].unsqueeze(-1),  # [N, 1] - 抽屉位置
                self._cabinet.data.joint_vel[:, 3].unsqueeze(-1),  # [N, 1] - 抽屉速度
                finger_positions,  # [N, 15] - 手指相对位置
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        drawer_pos = self._cabinet.data.body_pos_w[env_ids, self.drawer_link_idx]
        drawer_rot = self._cabinet.data.body_quat_w[env_ids, self.drawer_link_idx]
        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
            self.drawer_grasp_rot[env_ids],
            self.drawer_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
            drawer_rot,
            drawer_pos,
            self.drawer_local_grasp_rot[env_ids],
            self.drawer_local_grasp_pos[env_ids],
        )

    def _compute_rewards(
        self,
        actions,
        cabinet_dof_pos,
        franka_grasp_pos,
        drawer_grasp_pos,
        franka_grasp_rot,
        drawer_grasp_rot,
        franka_lfinger_pos,
        franka_rfinger_pos,
        gripper_forward_axis,
        drawer_inward_axis,
        gripper_up_axis,
        drawer_up_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        open_reward_scale,
        action_penalty_scale,
        finger_reward_scale,
        joint_positions,
    ):
        # distance from hand to the drawer
        d = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
        axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

        dot1 = (
            torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of forward axis for gripper
        dot2 = (
            torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        # how far the cabinet has been opened out
        open_reward = cabinet_dof_pos[:, 3]  # drawer_top_joint

        # penalty for distance of each finger from the drawer handle
        lfinger_dist = franka_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2]
        rfinger_dist = drawer_grasp_pos[:, 2] - franka_rfinger_pos[:, 2]
        finger_dist_penalty = torch.zeros_like(lfinger_dist)
        finger_dist_penalty += torch.where(lfinger_dist < 0, lfinger_dist, torch.zeros_like(lfinger_dist))
        finger_dist_penalty += torch.where(rfinger_dist < 0, rfinger_dist, torch.zeros_like(rfinger_dist))

        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + open_reward_scale * open_reward
            + finger_reward_scale * finger_dist_penalty
            - action_penalty_scale * action_penalty
        )

        self.extras["log"] = {
            "dist_reward": (dist_reward_scale * dist_reward).mean(),
            "rot_reward": (rot_reward_scale * rot_reward).mean(),
            "open_reward": (open_reward_scale * open_reward).mean(),
            "action_penalty": (-action_penalty_scale * action_penalty).mean(),
            "left_finger_distance_reward": (finger_reward_scale * lfinger_dist).mean(),
            "right_finger_distance_reward": (finger_reward_scale * rfinger_dist).mean(),
            "finger_dist_penalty": (finger_reward_scale * finger_dist_penalty).mean(),
        }

        # bonus for opening drawer properly
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.25, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + 0.25, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.35, rewards + 0.25, rewards)

        return rewards

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos
