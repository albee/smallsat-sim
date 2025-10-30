import os
import time
import jax
import jax.numpy as jnp
from flax import nnx

from smallsat_sim.envs.vec_env import VecEnv
from smallsat_sim.planners.base_planner import BasePlanner
from smallsat_sim.controllers.pd.vectorized_controller import VectorizedPDController
from smallsat_sim.controllers.rl.algorithms.vpg import VPG
from smallsat_sim.controllers.rl.algorithms.ppo import PPO
from smallsat_sim.controllers.rl.modules.adaptation_module import AdaptationModule
from smallsat_sim.controllers.rl.runners.runner_utils import load_trained_modules
from smallsat_sim.utils.helpers_jax import normalize_obs


class RLController(object):
    """
    RL controller that uses the saved actor weights.
    """

    def __init__(
        self, env: VecEnv, planner: BasePlanner, ckpt_name: str | None = None
    ) -> None:
        # Initialize the environment and agent
        self.env = env
        self.planner = planner
        base_key = self.env.next_rng_keys(1)[0]
        self._rng, agent_key = jax.random.split(base_key)
        self.agent = PPO(self.env, planner, rng_key=agent_key)
        self.am = AdaptationModule(50, env.obs_dim + env.act_dim, env.ext_dim)
        self.pd_ctrl = VectorizedPDController(env, planner)

        # Deployment length (if applicable)
        self.deployment_len = self.env.env_cfg.control.RL.deployment_len

        # Path to the saved checkpoints
        self.ckpt_dir = "src/smallsat_sim/controllers/rl/checkpoints/"
        if ckpt_name is not None:
            self.ckpt_filename = ckpt_name
        else:
            self._get_training_state_file_name()

    def _take_keys(self, count: int = 1):
        """
        Consume ``count`` RNG keys from the runner seed.
        """
        if count < 1:
            raise ValueError("count must be >= 1")
        splits = jax.random.split(self._rng, count + 1)
        self._rng = splits[0]
        if count == 1:
            return splits[1]
        return splits[1:]

    def control(
        self,
        stage: str = "deployment",
        phase: int = 2,
        test_pd: bool = False,
        perturbation_distribution: jnp.ndarray | None = None,
        apply_disturbances: bool = False,
    ) -> None:
        """
        Control the agent using the previously trained RL controller.
        If phase == 1, use extrinsics computed in sim.
        If phase == 2, use extrinsics computed by adaptation module.
        """
        if not test_pd:
            # Check if trained actor, critic and adaptation modules are available and load them
            file_path = os.path.join(self.ckpt_dir, self.ckpt_filename)
            if os.path.isfile(file_path):
                restored_state = load_trained_modules(self.ckpt_dir, self.ckpt_filename)
                nnx.update(
                    self.agent.actor.mu_net, restored_state["actor_model"].mu_net
                )
                nnx.update(
                    self.agent.critic.v_net, restored_state["critic_model"].v_net
                )
                if self.env.use_adaptive_approach and phase == 2:
                    adapt_module_state = load_trained_modules(
                        self.ckpt_dir, "adapt_module_state.pkl"
                    )
                    nnx.update(self.am, adapt_module_state["am_model"])
            else:
                raise Exception("Not all necessary modules have been trained yet.")

        # Vectorize adaptation module
        self.adaptation_module = jax.vmap(self.am)

        # Helper variables to normalize the observations
        num_saved_obs = 1000
        last_obs = jnp.zeros((self.env.num_envs, num_saved_obs, self.env.obs_dim))
        step = 0

        self.env.reset()
        self.env.reset_perturbations()
        start_time = time.time()
        next_waypoint = self.planner.get_reference(self.env.get_obs())
        states = self.env.get_states(next_waypoint)
        states_normalized = states

        # No history in the beginning
        state_action_history = jnp.zeros(
            (self.env.num_envs, 50, self.env.obs_dim + self.env.act_dim)
        )

        # Initialize residuals
        if self.env.use_adaptive_approach is True:
            res = jnp.zeros(self.env.res_dim)
        else:
            res = jnp.empty((self.env.num_envs, 0))

        terminal = jnp.zeros(self.env.num_envs, dtype=bool)

        while (
            True and step < self.deployment_len
            if self.deployment_len is not None
            else True
        ):
            real_time = time.time() - start_time
            sim_time = self.env.mjx_batch.time[0]

            # Start perturbations after 100 steps
            if self.deployment_len >= 100 and step == 100:
                if perturbation_distribution is not None:
                    self.env.apply_random_perturbations(
                        key=self._take_keys(),
                        fraction_perturbed_envs=1.0,
                        perturbation_distribution=perturbation_distribution,
                    )
                elif apply_disturbances:
                    self.env.apply_random_disturbance(
                        key=self._take_keys(),
                        fraction_disturbed_envs=1.0,
                    )

            if hasattr(self.env, "renderer") and self.env.renderer is not None:
                self.env._visualize_renderer(self.planner.reference_point_list)

            if not test_pd:
                actions = self.agent.get_control_input(
                    stage,
                    jnp.concatenate([states, res], axis=1),  # Use un-normalized states
                )  # No sampling/exploration noise needed
            else:
                actions = jnp.asarray(
                    self.pd_ctrl.get_control_input(self.env, next_waypoint)
                )

            # Update state-action history
            state_action = jnp.expand_dims(
                jnp.concatenate([states, actions], axis=1), axis=1
            )
            state_action_history = jnp.concatenate(
                [state_action_history[:, 1:, :], state_action], axis=1
            )

            # Update next waypoint and states
            next_waypoint = self.planner.get_reference(self.env.get_obs())
            states = self.env.get_states(next_waypoint)
            if step == (num_saved_obs):
                last_obs = jnp.roll(last_obs, num_saved_obs - 1, axis=1)
            last_obs = last_obs.at[:, step, :].set(states)
            states_normalized = normalize_obs(states, last_obs, step)

            # Step the environment
            _, _ = self.env.transition(actions, states, next_waypoint)
            step += 1

            # Compute extrinsics and residuals
            actual_wrench = self.env.get_actual_wrench()
            if self.env.use_adaptive_approach is True:
                if phase == 1:
                    ext = actual_wrench
                elif phase == 2:
                    ext = self.adaptation_module(state_action_history)
                else:
                    raise Exception("There only exist two training phases.")
                desired_wrench = self.env.get_desired_wrench(actions)
                res = ext - desired_wrench
            else:
                res = jnp.empty((self.env.num_envs, 0))

            # Log data
            self.agent._log(
                self.env.run_id,
                float(self.env.mjx_batch.time[0]),
                stage,
                self.env,
                actual_wrench,
                ext,
            )

            if self.planner.completed_path.all():
                break

    def _get_training_state_file_name(self) -> None:
        """
        Set the checkpoint file name for the training state.
        """
        parts = ["training_state"]

        if self.env.use_adaptive_approach:
            parts.append("adaptive")
        if self.env.use_pretrained:
            parts.append("pretrained")
        if not self.env.train_with_failures:
            parts.append("nominal")

        self.ckpt_filename = "_".join(parts) + ".pkl"
