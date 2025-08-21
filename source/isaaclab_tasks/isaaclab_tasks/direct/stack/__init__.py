# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stacking environments for direct RL."""

import gymnasium as gym

from . import agents
from .stack_env import StackEnv, StackEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Stack-Jaka-Direct-v0",
    entry_point="isaaclab_tasks.direct.stack:StackEnv",
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.stack.stack_env:JakaStackEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StackPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
