from abc import ABC, abstractmethod
from typing import Optional
import jax
import jax.numpy as jnp
from flax import nnx

from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.controllers.base_controller import BaseController
from smallsat_sim.controllers.pd.vectorized_controller import VectorizedPDController
from smallsat_sim.controllers.rl.modules.base_network import Critic
from smallsat_sim.controllers.rl.modules.base_policy import Actor


class BaseAgent(BaseController):
    """
    Base implementation for actor-critic agents.
    """

    def __init__(self, env, planner, activation=nnx.tanh) -> None:
        self.ctrl_cfg = env.env_cfg.control.RL
        super().__init__(env, planner, self.ctrl_cfg)

        self.ctrl_cfg = env.env_cfg.control.RL
        self.env = env

        self.pd_ctrl = VectorizedPDController(env, planner)

        self.num_layers = 2
        self.layer_width = 64
        hidden_sizes = [self.layer_width] * self.num_layers
        self.actor = Actor(env.obs_dim, env.act_dim, hidden_sizes, activation)
        self.critic = Critic(env.obs_dim, hidden_sizes, activation)

    def act(
        self, states: jnp.ndarray, epoch: Optional[int] = None
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Return actions, value functions, and log-likelihood of chosen actions for given states.
        """
        pi, _ = self.actor.forward(states)
        key = jax.random.PRNGKey(42)
        actions = pi.sample(seed=key)
        actions = self.bootstrap_policies(actions, "continuous", epoch)
        values = self.critic.forward(states)
        logp = self.actor._log_prob_from_dist(pi, actions)

        return actions, values, logp

    def get_control_input(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate the control input based on current observations for each environment.
        """
        return self.act(obs)[0]
    
    def bootstrap_policies(self, actions:jnp.ndarray, strategy: Optional[str] = "sequential", epoch: Optional[int] = None) -> jnp.ndarray:
        """
        Overwrite (parts of) the actions with PD control input.
        """

        self.env.obs = self.env.get_obs()
        ctrl_input = self.pd_ctrl.get_control_input(self.env)
        pd_actions = jnp.asarray(ctrl_input)
        
        if strategy == "sequential":
            if epoch is not None and epoch < 10: # Number of training epochs tuned empirically
                actions = pd_actions
        elif strategy == "continuous":
            # frac = jnp.max()
            actions = pd_actions # TODO
        else:
            raise Exception("This bootstrapping strategy has not been implemented. Options are [sequential] and [continuous].")
        
        return actions
        
    @abstractmethod
    def update_policy_gradient(
        self,
        actor_lr: float,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        tdres: jnp.ndarray,
        logp: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Update the policy gradient. Return the actor loss.
        """
        pass

    @abstractmethod
    def update_value_function(
        self, critic_lr: float, obs: jnp.ndarray, returns: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Update the value function. Return the critic loss.
        """
        pass

    def _log(self, run_id: int, timestamp: float, env: BaseEnv) -> None:
        """
        Logs desired quantities if flag is enabled
        """
        raise NotImplementedError(
            f"The _log method is not implemented for the class {self.__class__.__name__}"
        )
