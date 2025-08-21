# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns,ImuCfg    
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots.mdog import Mit_CFG # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.8),
            "dynamic_friction_range": (0.8, 1.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.3,1.3),
            "operation": "add",
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.6, 1.2),
            "damping_distribution_params": (0.8, 1.5),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    randomize_rigid_body_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "inertia_distribution_params": (0.3, 1.2),
            "operation": "scale",
        },
    )

    randomize_com_positions = EventTerm(
        func=mdp.randomize_com_positions,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_distribution_params": (-0.1, 0.1),
            "operation": "add",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(7, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}, #,,"roll": (-1,1),"pitch": (-1,1), "yaw": (-1, 1),"z": (-0.5, 0.5)
    )

    # randomize_apply_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "force_range": (-10.0, 10.0),
    #         "torque_range": (-10.0, 10.0),
    #     },
    # )

    randomize_reset_joints = EventTerm(
        # func=mdp.reset_joints_by_scale,
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # randomize_reset_base = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14),
    #                                        "roll": (-3.14, 3.14),
    #             "pitch": (-3.14, 3.14),
    #             "yaw": (-3.14, 3.14),
    #             },#
    #         "velocity_range": {
    #             "x": (-0.5, 0.5),
    #             "y": (-0.5, 0.5),
    #             "z": (-0.5, 0.5),
    #             "roll": (-0.5, 0.5),
    #             "pitch": (-0.5, 0.5),
    #             "yaw": (-0.5, 0.5),
    #         },
    #     },
    # )

    


@configclass
class MitFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s =20
    decimation = 8
    action_scale = [0.125,0.125,0.125 ,0.125 ,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25]  # scale for each action
    #action_scale = [scale * 0.0 for scale in action_scale]  # scale down to 0.1
    action_space = 12
    observation_space = 48 +64
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 400,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
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
        debug_vis=False,
    )



    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = Mit_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    imu: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/imu",  # 将IMU安装在机器人基座上
        update_period=0.005,  # 更新频率，通常与仿真步长相同
    )
      # reward scales
    # lin_vel_reward_scale = 2
    # yaw_rate_reward_scale = 1
    # z_vel_reward_scale = -2.0
    # ang_vel_reward_scale = -0.05
    # joint_torque_reward_scale = -3e-5
    # joint_accel_reward_scale = -2.5e-7
    # action_rate_reward_scale = -0.01
    # feet_air_time_reward_scale =1.5
    # undesired_contact_reward_scale = -0
    # flat_orientation_reward_scale = -5.0
    upward_reward_scale = 0
    # joint_pos_limits_reward_scale = -1.0
    joint_vel_reward_scale = -1e-3
    stand_still = 2

    lin_vel_reward_scale =1.5
    yaw_rate_reward_scale = 0.75
    z_vel_reward_scale = -2
    ang_vel_reward_scale = -0.01
    joint_torque_reward_scale = -2.5e-7
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    action_acceleration_reward_scale = -0.01  # Penalty for action acceleration changes
    feet_air_time_reward_scale = -0.5  # 
    undesired_contact_reward_scale = -1
    flat_orientation_reward_scale = -0
    height_tracking_scale =1.5    
    long_contact_penalty_scale = -0  # 长时间接触地面时的惩罚
    joint_direction_change_scale = -0.01  # 惩罚关节方向的变化
    joint_pos_deviation_scale = -0.0
    joint_power_reward_scale: float = -0.0001
    feet_tangential_force_reward_scale: float = -0.01  # 脚部切向力惩罚比例（基于绝对大小）

    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

@configclass
class MitRoughEnvCfg(MitFlatEnvCfg):
    # env
    #observation_space = 48 + 64
    observation_space = 235
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0
