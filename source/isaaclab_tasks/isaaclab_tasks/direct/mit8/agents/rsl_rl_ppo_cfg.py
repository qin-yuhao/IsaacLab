# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class mitFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 400
    save_interval = 50
    experiment_name = "mit_flat_directv8"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )

    # policy = RslRlPpoActorCriticCfg(
    #     class_name="ResidualActorCritic",
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[128 for _ in range(12)],
    #     critic_hidden_dims=[128 for _ in range(12)],
    #     activation="elu",
    # )

    algorithm = RslRlPpoAlgorithmCfg(
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
    )


@configclass
class mitRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "mit_rough_directv8"
    empirical_normalization = False

    # policy = RslRlPpoActorCriticCfg(
    #     class_name="ResidualActorCritic",
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[256 for _ in range(12)],
    #     critic_hidden_dims=[256 for _ in range(12)],
    #     activation="elu",
    # )
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256 for _ in range(3)],
        critic_hidden_dims=[256 for _ in range(3)],
        activation="elu",
    )
    # algorithm = RslRlPpoAlgorithmCfg(
    #     value_loss_coef=1.0,
    #     use_clipped_value_loss=True,
    #     clip_param=0.2,
    #     entropy_coef=0.005,
    #     num_learning_epochs=5,
    #     num_mini_batches=4,
    #     learning_rate=1.0e-3,
    #     schedule="adaptive",
    #     gamma=0.99,
    #     lam=0.95,
    #     desired_kl=0.01,
    #     max_grad_norm=1.0,
        
    # )

    algorithm = RslRlPpoAlgorithmCfg(
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
)
