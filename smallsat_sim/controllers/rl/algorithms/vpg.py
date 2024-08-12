import jax
import jax.numpy as jnp
from flax import nnx

from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.controllers.base_controller import BaseController
from smallsat_sim.controllers.rl.modules.base_network import Critic
from smallsat_sim.controllers.rl.modules.base_policy import Actor


class VPGAgent(BaseController):
    """
    Base agent (Vanilla Policy Gradient with Generalized Advantage Estimation).
    """

    def __init__(self, env, planner, activation=nnx.tanh) -> None:
        self.ctrl_cfg = env.env_cfg.control.RL
        super().__init__(env, planner, self.ctrl_cfg)

        self.ctrl_cfg = env.env_cfg.control.RL
        self.env = env

        self.num_layers = 2
        self.layer_width = 64
        hidden_sizes = [self.layer_width] * self.num_layers
        self.actor = Actor(env.obs_dim, env.act_dim, hidden_sizes, activation)
        self.critic = Critic(env.obs_dim, hidden_sizes, activation)

    def act(self, states: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Return actions, value functions, and log-likelihood of chosen actions for given states.
        """
        # TODO: disable gradient computation with states = jax.lax.stop_gradient(states)?
        pi, _ = self.actor.forward(states)
        key = jax.random.PRNGKey(42)
        actions = pi.sample(seed=key)
        values = self.critic.forward(states)
        logp = self.actor._log_prob_from_dist(pi, actions)

        return actions, values, logp

    def get_control_input(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate the control input based on current observations for each environment.
        """
        return self.act(obs)[0]

    def _log(self, run_id: int, timestamp: float, env: BaseEnv) -> None:
        """
        Logs desired quantities if flag is enabled
        """
        raise NotImplementedError(
            f"The _log method is not implemented for the class {self.__class__.__name__}"
        )
