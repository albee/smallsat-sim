import torch
import torch.nn as nn

from smallsat_sim.controllers.rl.modules.mlp import mlp


class Actor(nn.Module):
    """
    The policy network. Inspired from https://spinningup.openai.com/en/latest/algorithms/vpg.html.
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation) -> None:
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs: torch.tensor):
        """
        Return a distribution over actions given observations.
        """
        return torch.distributions.categorical.Categorical(logits=self.logits_net(obs)) # TODO: adapt to batched envs here
    
    def _log_prob_from_dist(self, pi: torch.tensor, actions: torch.tensor):
        """
        Return the log-probability of actions under the action distribution.
        """
        return pi.log_prob(actions)

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
    