import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from smallsat_sim.controllers.base_controller import BaseController
from smallsat_sim.controllers.rl.modules.base_network import Critic
from smallsat_sim.controllers.rl.modules.base_policy import Actor

class VPGAgent(BaseController):
    """
    Base agent (Vanilla Policy Gradient with Generalized Advantage Estimation).
    """
    def __init__(self, env, planner, activation=nn.Tanh) -> None:
        super().__init__(env, planner)

        self.ctrl_cfg = env.env_cfg.control.RL
        self.env = env

        self.num_layers = 2
        self.layer_width = 64
        hidden_sizes = [self.layer_width] * self.num_layers
        self.actor = Actor(env.obs_dim, env.act_dim, hidden_sizes, activation)
        self.critic = Critic(env.obs_dim, hidden_sizes, activation)

    def act(self, states: torch.tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Return actions, value functions, and log-likelihood of chosen actions for given states.
        """
        with torch.no_grad():
            pi = self.actor.forward(states)
            actions = pi.sample()

            value_funcs = self.critic.forward(states)

            logp = self.actor._log_prob_from_dist(pi, actions)

        return actions, value_funcs, logp
        
    def get_control_input(self, obs: torch.tensor) -> np.ndarray:
        """
        Calculate the control input based on current observations for each environment.
        """
        ctrl_inputs = self.act(obs)[0]

        return ctrl_inputs.detach().numpy()