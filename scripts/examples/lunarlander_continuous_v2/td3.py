# -*- coding: utf-8 -*-
"""Run module for TD3 on LunarLanderContinuous-v2.

- Author: whikwon
- Contact: whikwon@gmail.com
"""

import argparse

import gym
import torch
import torch.optim as optim

from algorithms.common.networks.mlp import MLP
from algorithms.common.noise import GaussianNoise
from algorithms.td3.agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {
    "GAMMA": 0.99,
    "TAU": 1e-3,
    "BUFFER_SIZE": int(1e5),
    "BATCH_SIZE": 128,
    "LR_ACTOR": 1e-3,
    "LR_CRITIC": 1e-3,
    "WEIGHT_DECAY": 1e-6,
    "EXPLORATION_NOISE": 0.2,
    "TARGET_POLICY_NOISE": 0.2,
    "TARGET_POLICY_NOISE_CLIP": 0.5,
    "POLICY_UPDATE_FREQ": 2
}


def run(env: gym.Env, args: argparse.Namespace, state_dim: int, action_dim: int):
    """Run training or test.

    Args:
        env (gym.Env): openAI Gym environment with continuous action space
        args (argparse.Namespace): arguments including training settings
        state_dim (int): dimension of states
        action_dim (int): dimension of actions

    """
    hidden_sizes = [256, 256]

    # create actor
    actor = MLP(
        input_size=state_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes,
        output_activation=torch.tanh,
    ).to(device)

    actor_target = MLP(
        input_size=state_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes,
        output_activation=torch.tanh,
    ).to(device)
    actor_target.load_state_dict(actor.state_dict())

    # create critic1
    critic1 = MLP(
        input_size=state_dim + action_dim, output_size=1, hidden_sizes=hidden_sizes
    ).to(device)

    critic_target1 = MLP(
        input_size=state_dim + action_dim, output_size=1, hidden_sizes=hidden_sizes
    ).to(device)
    critic_target1.load_state_dict(critic1.state_dict())

    # create critic2
    critic2 = MLP(
        input_size=state_dim + action_dim, output_size=1, hidden_sizes=hidden_sizes
    ).to(device)

    critic_target2 = MLP(
        input_size=state_dim + action_dim, output_size=1, hidden_sizes=hidden_sizes
    ).to(device)
    critic_target2.load_state_dict(critic2.state_dict())

    # create optimizer
    actor_optim = optim.Adam(
        actor.parameters(),
        lr=hyper_params["LR_ACTOR"],
        weight_decay=hyper_params["WEIGHT_DECAY"],
    )

    critic_optim1 = optim.Adam(
        critic1.parameters(),
        lr=hyper_params["LR_CRITIC"],
        weight_decay=hyper_params["WEIGHT_DECAY"],
    )

    critic_optim2 = optim.Adam(
        critic2.parameters(),
        lr=hyper_params["LR_CRITIC"],
        weight_decay=hyper_params["WEIGHT_DECAY"],
    )

    # noise
    exploration_noise = GaussianNoise(
        action_dim,
        args.seed,
        mu=0,
        sigma=hyper_params["EXPLORATION_NOISE"]
    )
    target_policy_noise = GaussianNoise(
        action_dim,
        args.seed,
        mu=0,
        sigma=hyper_params["TARGET_POLICY_NOISE"]
    )

    # make tuples to create an agent
    models = (actor, actor_target, critic1, critic_target1, critic2, critic_target2)
    optims = (actor_optim, critic_optim1, critic_optim2)
    noises = (exploration_noise, target_policy_noise)

    # create an agent
    agent = Agent(
        env,
        args,
        hyper_params,
        models,
        optims,
        noises
    )

    # run
    if args.test:
        agent.test()
    else:
        agent.train()