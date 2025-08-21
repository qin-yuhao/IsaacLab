# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for HIM PPO algorithm for AnymalC environments."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_rl.rsl_rl.him_cfg import RslRlHimActorCriticCfg, RslRlHimPpoAlgorithmCfg


@configclass
class AnymalCFlatHIMPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for HIM PPO with AnymalC on flat terrain."""

    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 100
    experiment_name = "anymal_c_flat_him_direct"
    empirical_normalization = False
    
    policy = RslRlHimActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        # HIM specific parameters
        history_length=10,
        num_one_step_obs=48,  # flat terrain base observation size
        estimator_hidden_dims=[128, 128],
        estimator_output_dim=19,  # 3 for velocity + 16 for latent
        proto_dim=256,
        num_prototypes=512,
        tau=0.1,
        estimator_lr=1e-3,
    )
    
    algorithm = RslRlHimPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # HIM specific loss coefficients
        estimation_loss_coef=1.0,
        swap_loss_coef=0.5,
    )


@configclass
class AnymalCRoughHIMPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for HIM PPO with AnymalC on rough terrain."""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "anymal_c_rough_him_direct"
    empirical_normalization = False
    
    policy = RslRlHimActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # HIM specific parameters
        history_length=10,
        num_one_step_obs=235,  # rough terrain base observation size
        estimator_hidden_dims=[256, 128],  # Larger for complex terrain
        estimator_output_dim=19,  # 3 for velocity + 16 for latent
        proto_dim=256,
        num_prototypes=512,
        tau=0.1,
        estimator_lr=1e-3,
    )
    
    algorithm = RslRlHimPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # HIM specific loss coefficients
        estimation_loss_coef=1.0,
        swap_loss_coef=0.5,
    )
