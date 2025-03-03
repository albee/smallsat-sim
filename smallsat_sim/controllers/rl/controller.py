import os
import time
import jax
import jax.numpy as jnp
from flax import nnx

from smallsat_sim.envs.vec_env import VecEnv
from smallsat_sim.planners.base_planner import BasePlanner
from smallsat_sim.controllers.rl.algorithms.vpg import VPG
from smallsat_sim.controllers.rl.algorithms.ppo import PPO
from smallsat_sim.controllers.rl.modules.adaptation_module import AdaptationModule
from smallsat_sim.controllers.rl.runners.runner_utils import load_trained_modules
from smallsat_sim.utils.helpers_jax import normalize_obs


class RLController(object):
    """
    RL controller that uses the saved actor weights.
    """

    def __init__(self, env: VecEnv, planner: BasePlanner) -> None:
        # Initialize the environment and agent
        self.env = env
        self.planner = planner
        self.agent = PPO(self.env, planner)
        self.am = AdaptationModule(50, env.obs_dim + env.act_dim, env.ext_dim)
        self.reference_points = planner.reference_points
        self.tracking_point_idx = 0

        # Path to save the checkpoints
        self.ckpt_dir = "smallsat_sim/controllers/rl/checkpoints/"
        self.ckpt_filename = "training_state.pkl"

    def control(
        self,
    ) -> None:
        """
        Control the agent using the previously trained RL controller.
        """
        # Check if trained actor, critic and adaptation modules are available and load them
        file_path = os.path.join(self.ckpt_dir, self.ckpt_filename)
        if os.path.isfile(file_path):
            restored_state = load_trained_modules(self.ckpt_dir, self.ckpt_filename)
            nnx.update(self.agent.actor.mu_net, restored_state["actor_model"].mu_net)
            nnx.update(self.agent.critic.v_net, restored_state["critic_model"].v_net)
            adapt_module_state = load_trained_modules(
                self.ckpt_dir, "adapt_module_state.pkl"
            )
            nnx.update(self.am, adapt_module_state["am_model"])
        else:
            raise Exception("No training has been done yet.")

        # Vectorize adaptation module
        self.adaptation_module = jax.vmap(self.am)

        # Helper variables to normalize the observations
        num_saved_obs = 1000
        last_obs = jnp.zeros((self.env.num_envs, num_saved_obs, self.env.obs_dim))
        step = 0

        self.env.reset()
        start_time = time.time()
        states = self.env.get_states(self.reference_points[self.tracking_point_idx])
        states_normalized = states

        state_action_history = jnp.zeros(
            (self.env.num_envs, 50, self.env.obs_dim + self.env.act_dim)
        )  # No history in the beginning
        ext = jnp.ones_like(self.env.mjx_batch.ctrl)  # No control input yet

        terminal = jnp.zeros(self.env.num_envs, dtype=bool)

        while True:
            real_time = time.time() - start_time
            sim_time = self.env.mjx_batch.time[0]
            if hasattr(self.env, "renderer") and self.env.renderer is not None:
                self.env._visualize_renderer(self.reference_points)
            actions = self.agent.get_control_input(
                jnp.concatenate([states, ext], axis=1)  # Use un-normalized states
            )  # No sampling/exploration noise needed
            states = self.env.get_states(self.reference_points[self.tracking_point_idx])
            if step == (num_saved_obs):
                last_obs = jnp.roll(last_obs, num_saved_obs - 1, axis=1)
            last_obs = last_obs.at[:, step, :].set(states)
            states_normalized = normalize_obs(states, last_obs, step)

            state_action = jnp.expand_dims(
                jnp.concatenate([states, actions], axis=1), axis=1
            )
            state_action_history = jnp.concatenate(
                [state_action_history[:, 1:, :], state_action], axis=1
            )

            ext = self.adaptation_module(state_action_history)

            _, terminal = self.env.transition(actions, states)
            step += 1
            if terminal.all():
                if self.tracking_point_idx == (len(self.reference_points) - 1):
                    break
                else:
                    self.tracking_point_idx += 1
