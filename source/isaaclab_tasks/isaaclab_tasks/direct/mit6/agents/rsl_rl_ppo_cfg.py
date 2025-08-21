# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticRecurrentCfg,RslRlDistillationStudentTeacherRecurrentCfg,RslRlDistillationAlgorithmCfg


@configclass
class MitFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 50
    experiment_name = "Mitv5_flat_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )

    # policy = RslRlPpoActorCriticRecurrentCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[128, 128, 128],
    #     critic_hidden_dims=[128, 128, 128],
    #     activation="elu",
    #     rnn_type="lstm",
    #     rnn_hidden_dim=128,
    #     rnn_num_layers=1,
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
class MitRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 13000
    save_interval = 50
    experiment_name = "mitv5_rough_direct"
    empirical_normalization = False


    # policy = RslRlPpoActorCriticCfg(
    #     #class_name="ResidualActorCritic",
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[512,256,128],
    #     critic_hidden_dims=[512,256,128],
    #     activation="elu",
    # )

    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )

    # policy = RslRlPpoActorCriticRecurrentCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[512, 256, 128],
    #     critic_hidden_dims=[512, 256, 128],
    #     activation="elu",
    #     rnn_type="lstm",
    #     rnn_hidden_dim=128,
    #     rnn_num_layers=1,
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






    # policy = RslRlDistillationStudentTeacherRecurrentCfg(
    #     init_noise_std=1.0,
    #     student_hidden_dims=[512, 256, 128],
    #     teacher_hidden_dims=[512, 256, 128],
    #     teacher_recurrent=False,
    #     activation="elu",
    #     rnn_type="lstm",
    #     rnn_hidden_dim=128,
    #     rnn_num_layers=1,
    # )

    # algorithm = RslRlDistillationAlgorithmCfg(
    #     num_learning_epochs=5,
    #     learning_rate=1.0e-4,
    #     gradient_length =  8,
    # )