# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Anymal-C-Direct-v0",
    entry_point=f"{__name__}.anymal_c_env:AnymalCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.anymal_c_env_cfg:AnymalCFlatEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Anymal-C-Direct-v0",
    entry_point=f"{__name__}.anymal_c_env:AnymalCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.anymal_c_env_cfg:AnymalCRoughEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

##
# Register HIM environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Anymal-C-HIM-Direct-v0",
    entry_point=f"{__name__}.anymal_c_him_env:AnymalCHIMEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.anymal_c_him_env_cfg:AnymalCFlatHIMEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_ppo_cfg:AnymalCFlatHIMPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Anymal-C-HIM-Direct-v0",
    entry_point=f"{__name__}.anymal_c_him_env:AnymalCHIMEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.anymal_c_him_env_cfg:AnymalCRoughHIMEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_ppo_cfg:AnymalCRoughHIMPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Anymal-C-HIM-Direct-Play-v0",
    entry_point=f"{__name__}.anymal_c_him_env:AnymalCHIMEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.anymal_c_him_env_cfg:AnymalCFlatHIMEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_ppo_cfg:AnymalCFlatHIMPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Anymal-C-HIM-Direct-Play-v0",
    entry_point=f"{__name__}.anymal_c_him_env:AnymalCHIMEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.anymal_c_him_env_cfg:AnymalCRoughHIMEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_ppo_cfg:AnymalCRoughHIMPPORunnerCfg",
    },
)
