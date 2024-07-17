import numpy as np
import torch
import torch.nn as nn

from smallsat_sim.controllers.rl.modules.mlp import mlp


class Actor(nn.Module):
    """
    The policy network. Inspired from https://spinningup.openai.com/en/latest/algorithms/vpg.html.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: int, activation, device) -> None:
        super().__init__()
        log_std = -0.5 * torch.ones(act_dim, dtype=torch.float32, device=device)
        self.log_std = torch.nn.Parameter(log_std)
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs: torch.tensor):
        """
        Return a Gaussian distribution over actions given observations.
        """
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return torch.distributions.normal.Normal(mu, std)
    
    def _log_prob_from_dist(self, pi: torch.tensor, actions: torch.tensor):
        """
        Return the log-probability of actions under the action distribution.
        """
        return pi.log_prob(actions).sum(axis=-1)

    def forward(self, obs: torch.tensor, actions=None):
        """
        Return action distributions for given observations and the log-likelihood of given actions under those distributions.
        """
        pi = self._distribution(obs)

        if actions is None:
            logp = None
        else:
            logp = self._log_prob_from_dist(pi, actions)

        return pi, logp
    