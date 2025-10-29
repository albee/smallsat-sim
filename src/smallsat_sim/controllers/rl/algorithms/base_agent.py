from abc import abstractmethod
import jax
import jax.numpy as jnp
from flax import nnx

from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.envs.vec_env import VecEnv
from smallsat_sim.controllers.base_controller import BaseController
from smallsat_sim.controllers.rl.modules.base_network import Critic
from smallsat_sim.controllers.rl.modules.base_policy import Actor


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
            env.ext_dim,
            act_low,
            act_high,
        )
        self.critic = Critic(env.obs_dim, hidden_sizes, activation, env.ext_dim)
        self.key = rng_key

    def act(
        self, states_ext: jnp.ndarray, log: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Return actions, value functions, and log-likelihood of chosen actions for given states.
        """
        pi, _ = self.actor.forward(states_ext)
        self.key, subkey = jax.random.split(self.key)
        pre_actions = pi.sample(seed=subkey)
        actions = self.actor.apply_action_bounds(pre_actions)
        values = self.critic.forward(states_ext)
        logp = jnp.asarray(self.actor._log_prob_from_dist(pi, actions))

        return actions, values, logp

    def get_control_input(
        self,
        env: BaseEnv,
        stage: str | None = None,
        obs_extrinsics: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """
        Calculate the control input based on current observations for each environment.
        """
        if stage is None or obs_extrinsics is None:
            raise ValueError("stage and obs_extrinsics must be provided")

        ctrl_input = self.actor.deterministic_action(obs_extrinsics)

        if stage != "am_training" and stage != "evaluation":
            self._log(
                self.env.run_id,
                float(self.env.mjx_batch.time[0]),
                stage,
                self.env,
                ctrl_input,
                obs_extrinsics,
            )

        return ctrl_input

    @abstractmethod
    def update_policy_gradient(
        self,
        key,
        obs_extrinsics: jnp.ndarray,
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
        obs_extrinsics: jnp.ndarray,
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
