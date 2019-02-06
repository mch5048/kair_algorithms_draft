# -*- coding: utf-8 -*-
"""Noise classes for baselines."""

import copy
import random

import numpy as np


class OUNoise:
    """Ornstein-Uhlenbeck process.

    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py
    """

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.state = np.float64(0.0)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

        random.seed(seed)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for _ in range(len(x))]
        )
        self.state = x + dx
        return self.state


class GaussianNoise:
    """Gaussian Noise process."""

    def __init__(self, size, seed, mu=0.0, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        """Sample Gaussian noise."""
        return np.random.normal(self.mu, self.sigma, size=self.size)
