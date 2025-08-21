# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import Dict

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab_assets.robots.jaka import JAKA_CFG


@configclass
class StackEnvCfg(DirectRLEnvCfg):
    """Configuration for the stacking environment."""
    
    # env
    episode_length_s = 5
    decimation = 2
    action_space = 8 
    observation_space = 42  # joints(7) + joint_vel(7) + eef_pos(3) + eef_quat(4) + gripper_pos(1) + cube_pos(9*3) + cube_quat(9*4) = 7 + 7 + 3 + 4 + 1 + 27 + 36 = 85, simplified to 42
    state_space = 0
    
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,  # 100Hz
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    
    # frame transformer for end-effector
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/Link_01",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/Link_06",
                name="ee_tcp",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1334],  # TCP offset from Link_06
                ),
            ),
        ],
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
    
    # reward scales
    dist_reward_scale = 0.6
    grasp_reward_scale = 1.0
    stack_reward_scale = 10.0
    action_penalty_scale = 0
    action_rate_penalty_scale = -0.01  # 新增action rate惩罚系数
    orientation_reward_scale = 1.0
    dof_acc_reward_scale = -2.5e-5  # Joint acceleration penalty
    dof_vel_reward_scale = -1e-2    # Joint velocity penalty
    
    # action scale
    action_scale = 7


@configclass
class JakaStackEnvCfg(StackEnvCfg):
    """Configuration for Jaka stacking environment."""
    
    # robot
    robot: ArticulationCfg = JAKA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    def __post_init__(self):
        super().__post_init__()
        # Set robot position and orientation - elevated so arm can reach table height
        # Table is at Z=0.4, table surface at ~0.85, so robot base should be higher
        self.robot.init_state.pos = (0.0, 0.0, 0.4)  # Raise robot base higher
        self.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)


class StackEnv(DirectRLEnv):
    """Direct RL environment for object stacking task."""
    
    cfg: StackEnvCfg
    
    def __init__(self, cfg: StackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        
        # Object tracking
        self.cube_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.grasp_progress = torch.zeros(self.num_envs, device=self.device)
        
        # Task progress tracking
        self.task_progress = torch.zeros(self.num_envs, device=self.device)  # 0: reaching, 1: grasping, 2: lifting, 3: stacking
        
        # Logging for episode rewards
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "reach_cube_2",
                "reach_cube_3",
                "grasp_cube_2",
                "grasp_cube_3",
                "stack_cube_2_on_1",
                "stack_cube_3_on_2",
                "action_penalty",
                "dof_acc_l2",
                "action_rate",
                "dof_vel_l2",
            ]
        }
        
    def _setup_scene(self):
        """Setup the scene entities."""
        # Robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        
        # End-effector frame transformer
        self._ee_frame = FrameTransformer(self.cfg.ee_frame)
        self.scene.sensors["ee_frame"] = self._ee_frame
        
        # Cubes with rigid body properties
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )
        
        # Cube 1 (blue - base)
        cube_1_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Cube_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.42), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=cube_properties,
            ),
        )
        
        # Cube 2 (red - to be stacked)
        cube_2_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.05, 0.42), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=cube_properties,
            ),
        )
        
        # Cube 3 (green - to be stacked)
        cube_3_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Cube_3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.60, -0.1, 0.42), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=cube_properties,
            ),
        )
        
        self._cube_1 = RigidObject(cube_1_cfg)
        self._cube_2 = RigidObject(cube_2_cfg)
        self._cube_3 = RigidObject(cube_3_cfg)
        
        self.scene.rigid_objects["cube_1"] = self._cube_1
        self.scene.rigid_objects["cube_2"] = self._cube_2
        self.scene.rigid_objects["cube_3"] = self._cube_3
        
        # Setup terrain (includes ground)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # Add table before cloning environments - it will be automatically cloned
        table_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        )
        table_cfg.func(
            "/World/envs/env_.*/Table",
            table_cfg,
            translation=(0.5, 0.0, 0.4),  # Place table 0.4m above ground
            orientation=(0.707, 0, 0, 0.707)
        )
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply actions before physics step."""
        self.actions = actions.clone()  # .clamp(-1.0, 1.0)
        # Use action_scale to scale actions and add to default joint positions
        self._processed_actions = self.cfg.action_scale * self.actions + self._robot.data.default_joint_pos
    
    def _apply_action(self):
        """Apply computed actions to robot."""
        self._robot.set_joint_position_target(self._processed_actions)
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check if episodes are done."""
        # Check if any cube has fallen (below table level)
        cube_1_fallen = self._cube_1.data.root_pos_w[:, 2] < 0.4  # Below table
        cube_2_fallen = self._cube_2.data.root_pos_w[:, 2] < 0.4
        cube_3_fallen = self._cube_3.data.root_pos_w[:, 2] < 0.4
        cubes_fallen = cube_1_fallen | cube_2_fallen | cube_3_fallen
        
        # Check if stacking is successful (cube_2 on cube_1 and cube_3 on cube_2)
        cube_2_on_1 = self._check_stacking(self._cube_2, self._cube_1)
        cube_3_on_2 = self._check_stacking(self._cube_3, self._cube_2)
        stacking_success = cube_2_on_1 & cube_3_on_2
        
        terminated = cubes_fallen | stacking_success
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        
        return terminated, truncated
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        # Get end-effector position
        ee_pos = self._get_ee_position()
        
        # Distance rewards to cubes
        cube_2_pos = self._cube_2.data.root_pos_w
        cube_3_pos = self._cube_3.data.root_pos_w
        
        # Reward for reaching cube_2 first
        dist_to_cube_2 = torch.norm(ee_pos - cube_2_pos, dim=-1)
        reach_reward_2 = 1.0 / (1.0 + dist_to_cube_2**2)
        
        # Reward for reaching cube_3
        dist_to_cube_3 = torch.norm(ee_pos - cube_3_pos, dim=-1)
        reach_reward_3 = 1.0 / (1.0 + dist_to_cube_3**2)
        
        # Grasp rewards
        grasp_reward_2 = self._compute_grasp_reward(cube_2_pos, ee_pos)
        grasp_reward_3 = self._compute_grasp_reward(cube_3_pos, ee_pos)
        
        # Stacking rewards
        stack_reward_2_on_1 = self._compute_stack_reward(self._cube_2, self._cube_1)
        stack_reward_3_on_2 = self._compute_stack_reward(self._cube_3, self._cube_2)
        
        # Action penalty
        action_penalty = torch.sum(self.actions**2, dim=-1)
        
        # Action rate penalty
        action_rate_penalty = torch.sum(torch.square(self.actions - getattr(self, '_previous_actions', torch.zeros_like(self.actions))), dim=-1)

        # Joint acceleration penalty
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        joint_vel_penalty = torch.sum(torch.square(self._robot.data.joint_vel), dim=1)
        
        # Create detailed rewards dictionary
        rewards = {
            "reach_cube_2": reach_reward_2 * self.cfg.dist_reward_scale * self.step_dt * 0,
            "reach_cube_3": reach_reward_3 * self.cfg.dist_reward_scale * self.step_dt,
            "grasp_cube_2": grasp_reward_2 * self.cfg.grasp_reward_scale * self.step_dt,
            "grasp_cube_3": grasp_reward_3 * self.cfg.grasp_reward_scale * self.step_dt,
            "stack_cube_2_on_1": stack_reward_2_on_1 * self.cfg.stack_reward_scale * self.step_dt,
            "stack_cube_3_on_2": stack_reward_3_on_2 * self.cfg.stack_reward_scale * self.step_dt,
            "action_penalty": action_penalty * self.cfg.action_penalty_scale * self.step_dt,
            "action_rate": action_rate_penalty * getattr(self.cfg, 'action_rate_penalty_scale', -0.01) * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.dof_acc_reward_scale * self.step_dt,
            "dof_vel_l2": joint_vel_penalty * self.cfg.dof_vel_reward_scale * self.step_dt,
        }
        
        # Combine total reward
        total_reward = (
            rewards["reach_cube_2"] + rewards["reach_cube_3"]
            + rewards["grasp_cube_2"] + rewards["grasp_cube_3"]
            + rewards["stack_cube_2_on_1"] + rewards["stack_cube_3_on_2"]
            + rewards["action_penalty"] + rewards["dof_acc_l2"]
        )
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        
        # Update previous actions
        self._previous_actions = self.actions.clone()
        
        return total_reward
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specific environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        
        # Reset robot
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.25, 0.25, (len(env_ids), self._robot.num_joints), self.device
        )
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        
        # Get robot root state and add terrain origins
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        # Write robot state
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Reset cubes using their default states with terrain origins
        # Cube 1
        default_cube_1_state = self._cube_1.data.default_root_state[env_ids].clone()
        default_cube_1_state[:, :3] += self._terrain.env_origins[env_ids]
        # Add small randomization
        rand_xy = sample_uniform(-0.02, 0.02, (len(env_ids), 2), self.device)
        default_cube_1_state[:, :2] += rand_xy
        self._cube_1.write_root_state_to_sim(default_cube_1_state, env_ids)
        
        # Cube 2
        default_cube_2_state = self._cube_2.data.default_root_state[env_ids].clone()
        default_cube_2_state[:, :3] += self._terrain.env_origins[env_ids]
        default_cube_2_state[:, :2] += rand_xy
        self._cube_2.write_root_state_to_sim(default_cube_2_state, env_ids)
        
        # Cube 3
        default_cube_3_state = self._cube_3.data.default_root_state[env_ids].clone()
        default_cube_3_state[:, :3] += self._terrain.env_origins[env_ids]
        default_cube_3_state[:, :2] += rand_xy
        self._cube_3.write_root_state_to_sim(default_cube_3_state, env_ids)
        
        # Reset task progress
        self.task_progress[env_ids] = 0
        self.cube_grasped[env_ids] = False
        self.grasp_progress[env_ids] = 0
        
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/cubes_fallen"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
    
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations."""
        # Robot joint positions and velocities (relative to default positions)
        dof_pos_scaled = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        dof_vel = self._robot.data.joint_vel * 0.1  # Scale velocities
        
        # End-effector position
        ee_pos = self._get_ee_position()
        
        # Cube positions and orientations (relative to robot base)
        robot_base_pos = self._robot.data.root_pos_w
        cube_1_pos_rel = self._cube_1.data.root_pos_w - robot_base_pos
        cube_2_pos_rel = self._cube_2.data.root_pos_w - robot_base_pos
        cube_3_pos_rel = self._cube_3.data.root_pos_w - robot_base_pos
        
        # Gripper state (approximated)
        gripper_pos = self._get_gripper_position()
        
        obs = torch.cat([
            dof_pos_scaled,
            dof_vel,
            ee_pos - robot_base_pos,  # Relative EE position
            cube_1_pos_rel,
            cube_2_pos_rel,
            cube_3_pos_rel,
            gripper_pos.unsqueeze(-1)
        ], dim=-1)
        
        return {"policy": torch.clamp(obs, -5.0, 5.0)}
    
    def _get_ee_position(self) -> torch.Tensor:
        """Get end-effector position using frame transformer."""
        # Use the frame transformer to get TCP position with offset
        return self._ee_frame.data.target_pos_w[..., 0, :]  # TCP position (first and only target frame)

    def _get_gripper_position(self) -> torch.Tensor:
        """Get gripper opening state."""
        # For Jaka, check finger joints
        finger_joints = self._robot.find_joints(".*finger.*")[0]
        return self._robot.data.joint_pos[:, finger_joints].mean(dim=1)
    
    def _compute_grasp_reward(self, object_pos: torch.Tensor, ee_pos: torch.Tensor) -> torch.Tensor:
        """Compute reward for grasping an object."""
        dist = torch.norm(ee_pos - object_pos, dim=-1)
        grasp_reward = torch.where(dist < 0.05, 1.0, 0.0)  # Close proximity bonus
        return grasp_reward
    
    def _compute_stack_reward(self, upper_object: RigidObject, lower_object: RigidObject) -> torch.Tensor:
        """Compute reward for stacking objects."""
        upper_pos = upper_object.data.root_pos_w
        lower_pos = lower_object.data.root_pos_w
        
        # Check horizontal alignment
        horizontal_dist = torch.norm(upper_pos[:, :2] - lower_pos[:, :2], dim=-1)
        
        # Check vertical stacking (upper should be above lower)
        vertical_dist = upper_pos[:, 2] - lower_pos[:, 2]
        
        # Reward when properly stacked
        stack_reward = torch.where(
            (horizontal_dist < 0.1) & (vertical_dist > 0.035) & (vertical_dist < 0.08),
            1.0, 0.0
        )
        
        return stack_reward
    
    def _check_stacking(self, upper_object: RigidObject, lower_object: RigidObject) -> torch.Tensor:
        """Check if upper object is stacked on lower object."""
        upper_pos = upper_object.data.root_pos_w
        lower_pos = lower_object.data.root_pos_w
        
        # Check alignment and height
        horizontal_dist = torch.norm(upper_pos[:, :2] - lower_pos[:, :2], dim=-1)
        vertical_dist = upper_pos[:, 2] - lower_pos[:, 2]
        
        stacked = (horizontal_dist < 0.05) & (vertical_dist > 0.035) & (vertical_dist < 0.08)
        return stacked
