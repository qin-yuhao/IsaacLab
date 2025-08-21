# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for AnymalC environment with HIM (History-Informed Model) support."""

from isaaclab.utils import configclass

from .anymal_c_env_cfg import AnymalCFlatEnvCfg, AnymalCRoughEnvCfg

##
# HIM-specific configs
##


@configclass
class AnymalCFlatHIMEnvCfg(AnymalCFlatEnvCfg):
    """Configuration for AnymalC flat environment with HIM support."""

    # HIM specific parameters
    history_length: int = 10
    """Number of observation steps to keep in history for HIM."""

    # Update observation space to include history
    # Base observation: 48, with history: 48 * 10 = 480
    observation_space = 48 * 10

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Ensure we have the right observation space for HIM
        self.observation_space = 48 * self.history_length


@configclass
class AnymalCRoughHIMEnvCfg(AnymalCRoughEnvCfg):
    """Configuration for AnymalC rough environment with HIM support."""

    # HIM specific parameters
    history_length: int = 10
    """Number of observation steps to keep in history for HIM."""

    # Update observation space to include history
    # Base observation: 235, with history: 235 * 10 = 2350
    observation_space = 235 * 10

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Ensure we have the right observation space for HIM
        self.observation_space = 235 * self.history_length


@configclass
class AnymalCFlatHIMEnvCfg_PLAY(AnymalCFlatHIMEnvCfg):
    """Configuration for AnymalC flat environment with HIM support for play/evaluation."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5


@configclass
class AnymalCRoughHIMEnvCfg_PLAY(AnymalCRoughHIMEnvCfg):
    """Configuration for AnymalC rough environment with HIM support for play/evaluation."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
