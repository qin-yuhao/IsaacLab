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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, CameraCfg,TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sim import FisheyeCameraCfg, PinholeCameraCfg
##
# Pre-defined configs
##
from isaaclab_assets.robots.mit import Mit_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG,TEST_TERRAINS_CFG # isort: skip


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (-5, 5),
            "velocity_range": (-10, 10),
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={  #roll 侧躺
            #"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14),"roll" : (-3.14, 3.14)},
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0, 0),"roll" : (-0, 0)},
            "velocity_range": {
                # "x": (-0.0, 0.0),
                # "y": (-0., 0.0),
                # "z": (-0.5, 0.5),
                # "roll": (-100, 100),
                # "pitch": (-0.5, 0.5),
                # "yaw": (-0.5, 0.5),
            },
        },
    )

    # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(1.0, 5.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5),"z": (1, 3),"roll": (-60, 60),"pitch": (-30, 30), "yaw": (-30, 30)}},
    # )


@configclass
class MitFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.25
    action_space = 12
    observation_space = 48  +2
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=1, replicate_physics=True) #,filter_collisions=False

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = Mit_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # reward scales
    # lin_vel_reward_scale = 1.5
    # yaw_rate_reward_scale = 0.
    lin_vel_cmd_reward_scale = 2
    ang_vel_cmd_reward_scale = 2
    approach_reward_scale = 0
    goal_reward_scale = 0
    stationary_penalty_scale = -10
    z_vel_reward_scale = -0.5
    ang_vel_reward_scale = -0.0
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -0
    heading_reward_scale = 0.5
    upward_reward_scale = 1
    speed_tracking_reward_scale = 2
    roll_recovery_scale: float = 0# 翻滚恢复奖励的权重
    foot_height_reward_scale: float = 0


@configclass
class MitRoughEnvCfg(MitFlatEnvCfg):
    # env
    #observation_space = 235 # 48 + 187
    # 48+64
    episode_length_s = 5
    #observation_space =  3 + 3 + 3 + 2 + 1 + 1 + 12 + 12  + 12 + 187
    observation_space =  3 + 3 + 3 + 2 + 1 + 1 + 12 + 12  + 12 + 64
    #observation_space = 390
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        #terrain_generator=ROUGH_TERRAINS_CFG,
        terrain_generator=TEST_TERRAINS_CFG,
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

    # mesh_prim_paths = "/World/FlatGrid"
    # terrain = TerrainImporterCfg(
    #     # prim_path="/World/ground",
    #     # terrain_type="plane",
    #     prim_path=mesh_prim_paths,
    #     terrain_type="usd",
    #     usd_path=r"C:\Users\Admin\Desktop\IsaacLab\terrain.usd",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    #     debug_vis=False,
    # )

    # we add a height scanner for perceptive locomotion
    # height_scanner = RayCasterCfg(
    #     prim_path="/World/envs/env_.*/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.8, 0.0, 20.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],#/World/envs/env_0/Robot/  ,"/World/envs/env_.*/Robot"
    # )

    front_lidar = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/lidar",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=False,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=8,vertical_fov_range=[-22.5, 22.5], horizontal_fov_range=[-22.5, 22.5],horizontal_res=5.625
        ),
        debug_vis=False, #/World/envs/env_0/Robot/base/collisions/base
        mesh_prim_paths=["/World/ground"],
        # mesh_prim_paths=["/World/ground", "/World/envs/env_.*/Robot/base/collisions/base",
        #                 "/World/envs/env_.*/Robot/FL_foot/collisions/FL_foot",
        #                 "/World/envs/env_.*/Robot/FR_foot/collisions/FR_foot",
        #                 "/World/envs/env_.*/Robot/RL_foot/collisions/RL_foot",
        #                 "/World/envs/env_.*/Robot/RR_foot/collisions/RR_foot",
        #                 "/World/envs/env_.*/Robot/FL_hip/collisions/FL_hip",
        #                 "/World/envs/env_.*/Robot/FR_hip/collisions/FR_hip",
        #                 "/World/envs/env_.*/Robot/RL_hip/collisions/RL_hip",
        #                 "/World/envs/env_.*/Robot/RR_hip/collisions/RR_hip",
        #                 "/World/envs/env_.*/Robot/FL_thigh/collisions/FL_thigh",
        #                 "/World/envs/env_.*/Robot/FR_thigh/collisions/FR_thigh",
        #                 "/World/envs/env_.*/Robot/RL_thigh/collisions/RL_thigh",
        #                 "/World/envs/env_.*/Robot/RR_thigh/collisions/RR_thigh",
        #                 "/World/envs/env_.*/Robot/FL_calf/collisions/FL_calf",
        #                 "/World/envs/env_.*/Robot/FR_calf/collisions/FR_calf",
        #                 "/World/envs/env_.*/Robot/RL_calf/collisions/RL_calf",
        #                 "/World/envs/env_.*/Robot/RR_calf/collisions/RR_calf",
        # ],
        max_distance=4,
    )

    # cam = TiledCameraCfg(
    #     prim_path="/World/envs/env_.*/Robot/lidar/cam",
    #     offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.499,0.499,-0.5001,-0.5001)), #变成了0.5 -0.5 0.5 0.5
    #     #update_period=0.005,
    #     width=640,
    #     height=480,
    #     debug_vis=False,
    #     spawn=PinholeCameraCfg(
    #         focal_length=2.8,
    #         horizontal_aperture=6.9,
    #     )
    # )

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0
