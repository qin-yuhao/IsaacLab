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
from isaaclab_assets.robots.devq import DEVQ_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.6),
            "dynamic_friction_range": (0.6, 1.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="interval",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
        interval_range_s=(5, 10),
    )

    random_push_by_set_velocity = EventTerm(
        func=mdp.push_by_setting_velocity,
        interval_range_s=(8, 10),
        mode="interval",
        params={
            "velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4), "z": (-0.1, 0.1)},
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
        },
    )

    # randomize_rigid_body_inertia = EventTerm(
    #     func=mdp.randomize_rigid_body_inertia,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "inertia_distribution_params": (0.5, 1.5),
    #         "operation": "scale",
    #     },
    # )

    randomize_reset_joints = EventTerm(
        # func=mdp.reset_joints_by_scale,
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (1, 1),
            "velocity_range": (0, 0),
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.1),
            "damping_distribution_params": (0.9, 1.2),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    randomize_reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-10, 10), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )









@configclass
class DevqFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20
    decimation = 4
    action_scale = 0.25
    action_space = 12
    observation_space = 48
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
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
        #terrain_type="usd",
        #usd_path=r"C:\Users\23655\Desktop\devq\test1.usd",
        terrain_type="plane",
        #terrain_generator=DEVQ_TERRAINS_CFG,
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=1, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = DEVQ_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    imu: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/imu",  # 将IMU安装在机器人基座上
        update_period=0.005,  # 更新频率，通常与仿真步长相同
    )

    # reward scales
    # lin_vel_reward_scale = 5
    # yaw_rate_reward_scale =1.25
    # z_vel_reward_scale = -2
    # ang_vel_reward_scale = -0.05
    # joint_torque_reward_scale = -2.5e-5
    # joint_accel_reward_scale = -5e-8
    # action_rate_reward_scale = -0.1
    # undesired_contact_reward_scale = -1
    # flat_orientation_reward_scale = -5.0
    # joint_pos_diff_reward_scale = -2
    # hip_deviation_penalty_scale = -2
    # thigh_angle_penalty_scale = -2
    # standing_pos_penalty_scale = -2
    # rear_leg_activity_scale = 0.5



    lin_vel_reward_scale = 1.8
    yaw_rate_reward_scale =0.7
    z_vel_reward_scale = -2
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -0.3e-3
    joint_accel_reward_scale = -1e-8
    action_rate_reward_scale = -0.008
    undesired_contact_reward_scale = -1
    flat_orientation_reward_scale = -5
    hip_deviation_penalty_scale = -0.3
    standing_pos_penalty_scale = -4
    feet_air_time_reward_scale = 1

    # lin_vel_reward_scale = 2
    # yaw_rate_reward_scale =0.7
    # z_vel_reward_scale = -0
    # ang_vel_reward_scale = -0.0
    # joint_torque_reward_scale = -0.0
    # joint_accel_reward_scale = -0.0
    # action_rate_reward_scale = -0.0
    # undesired_contact_reward_scale = -0.2
    # flat_orientation_reward_scale = -0.01
    # joint_pos_diff_reward_scale = -0.0
    # hip_deviation_penalty_scale = -0.0
    # thigh_angle_penalty_scale = -0.0
    # standing_pos_penalty_scale = -5
    # rear_leg_activity_scale = 0.0
    # feet_air_time_reward_scale = 0.5


@configclass
class DevqRoughEnvCfg(DevqFlatEnvCfg):
    # env 
    observation_space = 125 #235 = 48+187  77   11*7       187=11*17

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

    terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
    terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
    terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1, 0.6]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0
    episode_length_s = 50
