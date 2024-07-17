import torch
import torch.nn as nn

from smallsat_sim.controllers.rl.modules.mlp import mlp


class Critic(nn.Module):
    """
    The network used by the value function.
    """
    def __init__(self, obs_dim: int, hidden_sizes: int, activation, device) -> None:
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs: torch.tensor):
        """
        Return the value estimates for given observations.
        """
        return torch.squeeze(self.v_net(obs), -1)