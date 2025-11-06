from abc import abstractmethod
import jax
import jax.numpy as jnp
from flax import nnx

from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.envs.vec_env import VecEnv
from smallsat_sim.controllers.base_controller import BaseController
from smallsat_sim.controllers.rl.modules.base_network import Critic
from smallsat_sim.controllers.rl.modules.base_policy import Actor


def _sample_policy_from_modules(
    actor: Actor, critic: Critic, states_ext: jnp.ndarray, key: jnp.ndarray
):
    """
    Helper that samples actions, evaluates the critic, and computes log-probabilities.
    """
    pi, _ = actor.forward(states_ext)
    pre_actions = pi.sample(seed=key)
    actions = actor.apply_action_bounds(pre_actions)
    values = critic.forward(states_ext)
    logp = jnp.asarray(actor._log_prob_from_dist(pi, actions))
    return actions, values, logp


def policy_sample_from_state(
    actor_graphdef,
    actor_state,
    critic_graphdef,
    critic_state,
    states_ext: jnp.ndarray,
    key: jnp.ndarray,
):
    """
    Pure helper to sample policy outputs given actor/critic state containers.
    """
    actor = nnx.merge(actor_graphdef, actor_state)
    critic = nnx.merge(critic_graphdef, critic_state)
    actions, values, logp = _sample_policy_from_modules(actor, critic, states_ext, key)
    return actions, values, logp


class BaseAgent(BaseController):
    """
    Base implementation for actor-critic agents.
    """

    def __init__(
        self,
        env,
        planner,
        rng_key: jnp.ndarray,
        activation=nnx.tanh,
        adaptive=False,
    ) -> None:
        self.ctrl_cfg = env.env_cfg.control.RL
        super().__init__(env, planner, self.ctrl_cfg)

        self.ctrl_cfg = env.env_cfg.control.RL
        self.env = env
        self.planner = planner

        self.num_layers = 2
        self.layer_width = 64
        hidden_sizes = ([self.layer_width] * self.num_layers)[0]
        thruster_ranges = [
            thruster.forcerange for thruster in env.model_cfg.Thrusters.thruster_list
        ]
        act_low = jnp.asarray([fr[0] for fr in thruster_ranges], dtype=jnp.float32)
        act_high = jnp.asarray([fr[1] for fr in thruster_ranges], dtype=jnp.float32)
        self.actor = Actor(
            env.obs_dim,
            env.act_dim,
            hidden_sizes,
            activation,
            env.res_dim,
            act_low,
            act_high,
        )
        self.critic = Critic(env.obs_dim, hidden_sizes, activation, env.res_dim)
        self.key = rng_key
        self._sample_policy = nnx.jit(_sample_policy_from_modules, static_argnums=())
        self._actor_graphdef = nnx.graphdef(self.actor)
        self._critic_graphdef = nnx.graphdef(self.critic)
        self._policy_sample_fn = nnx.jit(
            policy_sample_from_state, static_argnums=(0, 2)
        )

    def act(
        self, states_ext: jnp.ndarray, log: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Return actions, value functions, and log-likelihood of chosen actions for given states.
        """
        self.key, subkey = jax.random.split(self.key)
        actions, values, logp = self._sample_policy(
            self.actor, self.critic, states_ext, subkey
        )

        return actions, values, logp

    def actor_critic_state(self):
        """
        Snapshot the actor and critic modules into immutable state containers.
        """
        return nnx.state(self.actor), nnx.state(self.critic)

    def functional_act(
        self,
        actor_state,
        critic_state,
        states_ext: jnp.ndarray,
        key: jnp.ndarray,
    ):
        """
        Pure interface for sampling using externally provided actor/critic states.
        """
        return self._policy_sample_fn(
            self._actor_graphdef,
            actor_state,
            self._critic_graphdef,
            critic_state,
            states_ext,
            key,
        )

    def get_control_input(
        self,
        stage: str | None = None,
        obs_residuals: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """
        Calculate the control input based on current observations for each environment.
        """
        if stage is None or obs_residuals is None:
            raise ValueError("stage and obs_residuals must be provided")

        ctrl_input = self.actor.deterministic_action(obs_residuals)

        return ctrl_input

    @abstractmethod
    def update_policy_gradient(
        self,
        key,
        obs_residuals: jnp.ndarray,
        actions: jnp.ndarray,
        tdres: jnp.ndarray,
        logp: jnp.ndarray,
        minibatch: bool = True,
    ) -> jnp.ndarray:
        """
        Update the policy gradient. Return the actor loss.
        """
        pass

    @abstractmethod
    def update_value_function(
        self,
        key,
        obs_residuals: jnp.ndarray,
        returns: jnp.ndarray,
        minibatch: bool = True,
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
