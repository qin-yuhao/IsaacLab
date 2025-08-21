# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AnymalC environment with HIM (History-Informed Model) support."""

from __future__ import annotations

import torch
from collections import deque

from .anymal_c_env import AnymalCEnv
from .anymal_c_him_env_cfg import AnymalCFlatHIMEnvCfg, AnymalCRoughHIMEnvCfg


class AnymalCHIMEnv(AnymalCEnv):
    """AnymalC environment with HIM support for history-based observations."""

    cfg: AnymalCFlatHIMEnvCfg | AnymalCRoughHIMEnvCfg

    def __init__(self, cfg: AnymalCFlatHIMEnvCfg | AnymalCRoughHIMEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # HIM-specific: Initialize observation history
        self._history_length = cfg.history_length
        # Determine base observation size (without history)
        if isinstance(cfg, AnymalCRoughHIMEnvCfg):
            self._base_obs_size = 235  # rough terrain observation size
        else:
            self._base_obs_size = 48   # flat terrain observation size

        # Initialize observation history buffer
        # Each element in history is [num_envs, base_obs_size]
        self._obs_history = deque(maxlen=self._history_length)
        
        # Initialize with zeros
        for _ in range(self._history_length):
            self._obs_history.append(torch.zeros(self.num_envs, self._base_obs_size, device=self.device))

    def _get_observations(self) -> dict:
        """Get observations with history for HIM algorithm."""
        # Get current base observation (same as parent)
        height_data = None
        if isinstance(self.cfg, AnymalCRoughHIMEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
        
        current_obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    height_data,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )

        # Add current observation to history
        self._obs_history.append(current_obs.clone())

        # Create history tensor: [num_envs, history_length * base_obs_size]
        history_tensor = torch.cat(list(self._obs_history), dim=-1)

        # HIM observations include:
        # 1. policy: current observation for immediate policy decisions
        # 2. critic: full history for value function
        # 3. next_critic: next step's full history (filled during rollout)
        observations = {
            "policy": current_obs,
            "critic": history_tensor,
            "next_critic": history_tensor  # This will be updated in the next step
        }
        return observations

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environment and clear history for reset environments."""
        super()._reset_idx(env_ids)
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Clear history for reset environments by setting to zeros
        for i in range(len(self._obs_history)):
            self._obs_history[i][env_ids] = 0.0

    def get_next_critic_observations(self) -> torch.Tensor:
        """Get the next critic observations for HIM algorithm."""
        # This is called after the environment step to get next observations
        # Return the current full history as next_critic observations
        return torch.cat(list(self._obs_history), dim=-1)
