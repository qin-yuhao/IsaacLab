# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for MIT Cheetah environment with HIM (History-Informed Model) support."""

from isaaclab.utils import configclass

from .mit_env_cfg import MitFlatEnvCfg, MitRoughEnvCfg

##
# HIM-specific configs
##


@configclass
class MitFlatHIMEnvCfg(MitFlatEnvCfg):
    """Configuration for MIT Cheetah flat environment with HIM support."""

    # HIM specific parameters
    history_length: int = 10
    """Number of observation steps to keep in history for HIM."""

    # Update observation space to include history
    # Base observation: 48 + 64 = 112, with history: 112 * 10 = 1120
    observation_space = (48 + 64) * 10

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Ensure we have the right observation space for HIM
        self.observation_space = (48 + 64) * self.history_length


@configclass
class MitRoughHIMEnvCfg(MitRoughEnvCfg):
    """Configuration for MIT Cheetah rough environment with HIM support."""

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
class MitFlatHIMEnvCfg_PLAY(MitFlatHIMEnvCfg):
    """Configuration for MIT Cheetah flat environment with HIM support for play/evaluation."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5


@configclass
class MitRoughHIMEnvCfg_PLAY(MitRoughHIMEnvCfg):
    """Configuration for MIT Cheetah rough environment with HIM support for play/evaluation."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # reduce the number of terrains to save memory
        if hasattr(self.terrain, 'terrain_generator') and self.terrain.terrain_generator is not None:
            self.terrain.terrain_generator.num_rows = 5
            self.terrain.terrain_generator.num_cols = 5
            self.terrain.terrain_generator.curriculum = False
