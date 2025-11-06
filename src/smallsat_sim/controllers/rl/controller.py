import os

import jax
import jax.numpy as jnp
from flax import nnx

from smallsat_sim.envs.vec_env import VecEnv, VecEnvStepOutput, vecenv_step
from smallsat_sim.planners.base_planner import BasePlanner
from smallsat_sim.controllers.pd.vectorized_controller import VectorizedPDController
from smallsat_sim.controllers.rl.algorithms.vpg import VPG
from smallsat_sim.controllers.rl.algorithms.ppo import PPO
from smallsat_sim.controllers.rl.modules.am_cnn import CNNAdaptationModule
from smallsat_sim.controllers.rl.modules.am_transformer import (
    TransformerAdaptationModule,
)
from smallsat_sim.controllers.rl.runners.runner_utils import load_trained_modules

_JITTED_VECENV_STEP = jax.jit(vecenv_step, static_argnames=("config",))


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
        self.state_action_dim = self.env.obs_dim + self.env.act_dim
        self.am = self._build_adaptation_module()
        self.pd_ctrl = VectorizedPDController(env, planner)

        # Deployment length (if applicable)
        self.deployment_len = self.env.env_cfg.control.RL.deployment_len

        # Path to the saved checkpoints
        self.ckpt_dir = "src/smallsat_sim/controllers/rl/checkpoints/"
        self.adaptation_module_file_name = self._get_adaptation_module_file_name()
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
                actor_state = restored_state["actor_model"]
                critic_state = restored_state["critic_model"]
                if isinstance(actor_state, dict):
                    nnx.update(self.agent.actor, actor_state)
                else:
                    nnx.update(self.agent.actor.mu_net, actor_state.mu_net)
                if isinstance(critic_state, dict):
                    nnx.update(self.agent.critic, critic_state)
                else:
                    nnx.update(self.agent.critic.v_net, critic_state.v_net)
                if self.env.use_adaptive_approach and phase == 2:
                    adapt_module_state = load_trained_modules(
                        self.ckpt_dir, self.adaptation_module_file_name
                    )
                    am_state = adapt_module_state["am_model"]
                    if isinstance(am_state, dict):
                        nnx.update(self.am, am_state)
                    else:
                        nnx.update(self.am, am_state)
            else:
                raise Exception("Not all necessary modules have been trained yet.")

        # Vectorize adaptation module
        self.adaptation_module = jax.vmap(self.am)

        self.env.reset()
        self.env.reset_perturbations()
        self.env._refresh_effect_states()

        vec_state = self.env.state_struct
        max_steps = (
            int(self.deployment_len)
            if self.deployment_len is not None
            else int(self.agent.max_ep_len)
        )
        next_waypoint = self.planner.get_reference(self.env.get_obs())
        states = self.env.get_states(next_waypoint)

        # No history in the beginning
        state_action_history = jnp.zeros(
            (self.env.num_envs, self.env.history_len, self.state_action_dim)
        )
        history_len = state_action_history.shape[1]
        history_counts = jnp.zeros(self.env.num_envs, dtype=jnp.int32)

        # Initialize residuals and extrinsics
        if self.env.use_adaptive_approach is True:
            res = jnp.zeros((self.env.num_envs, self.env.res_dim))
            ext = jnp.zeros((self.env.num_envs, self.env.ext_dim))
        else:
            res = jnp.empty((self.env.num_envs, 0))
            ext = jnp.empty((self.env.num_envs, 0))

        step = 0
        while True:
            if self.deployment_len is not None and step >= max_steps:
                break

            # Start perturbations after 100 steps
            if (
                self.deployment_len is not None
                and self.deployment_len >= 100
                and step == 100
            ):
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
                self.env._refresh_effect_states()

            if hasattr(self.env, "renderer") and self.env.renderer is not None:
                self.env._visualize_renderer(self.planner.reference_point_list)

            use_functional = not test_pd and stage not in {
                "deployment",
                "stuck_off_deployment",
                "stuck_on_deployment",
                "faulty_valve_deployment",
                "saturated_thrust_deployment",
                "thrust_instability_deployment",
                "constant_force_disturbances_deployment",
            }

            if use_functional:
                policy_input = (
                    jnp.concatenate([states, res], axis=1) if res.shape[-1] else states
                )
                actions = self.agent.get_control_input(
                    stage,
                    policy_input,
                )
                step_config = self.env.build_step_config(max_episode_len=max_steps)
                vec_state, step_output = _JITTED_VECENV_STEP(
                    vec_state, actions, next_waypoint, config=step_config
                )
                self.env.apply_state_struct(vec_state)
            else:
                if res.shape[-1]:
                    policy_input = jnp.concatenate([states, res], axis=1)
                else:
                    policy_input = states
                if test_pd:
                    prev_obs = self.env.get_obs()
                    actions = jnp.asarray(
                        self.pd_ctrl.get_control_input(self.env, next_waypoint)
                    )
                else:
                    actions = self.agent.get_control_input(stage, policy_input)
                rewards, terminals = self.env.transition(
                    actions,
                    states,
                    next_waypoint,
                )
                next_obs_batch = self.env.get_obs()
                next_states = self.env.get_states(next_waypoint)
                step_output = VecEnvStepOutput(
                    prev_states=states,
                    next_states=next_states,
                    rewards=rewards,
                    terminals=terminals,
                    commanded_ctrl=actions,
                    applied_ctrl=actions,
                    actual_wrench=self.env.get_actual_wrench(),
                    desired_wrench=self.env.get_desired_wrench(actions),
                    prev_obs=prev_obs if test_pd else next_obs_batch,
                    next_obs=next_obs_batch,
                    reward_components=self.env.get_last_reward_components(),
                )
                vec_state = self.env.state_struct

            # Update state-action history using pre-step states.
            state_action_history = jnp.roll(state_action_history, shift=-1, axis=1)
            state_action_history = state_action_history.at[:, -1, :].set(
                jnp.concatenate([states, actions], axis=1)
            )
            history_counts = jnp.minimum(history_counts + 1, history_len)

            # Prepare next-step state and waypoint from functional outputs.
            next_waypoint = self.planner.get_reference(step_output.next_obs)
            states = step_output.next_states

            step += 1

            actual_wrench = step_output.actual_wrench
            desired_wrench = step_output.desired_wrench
            if self.env.use_adaptive_approach:
                history_full = bool(jnp.all(history_counts >= history_len))
                if phase == 1:
                    ext = actual_wrench
                    res = ext - desired_wrench
                elif phase == 2:
                    if history_full:
                        ext = self.adaptation_module(state_action_history)
                        res = ext - desired_wrench
                else:
                    raise Exception("There only exist two training phases.")
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

            if step_output.terminals.all() or self.planner.completed_path.all():
                break

    def _build_adaptation_module(self):
        """
        Build the right adaptation module.
        """
        if self.env.am_architecture == "transformer":
            return TransformerAdaptationModule(
                self.env.history_len, self.state_action_dim, self.env.ext_dim
            )
        if self.env.am_architecture == "cnn":
            return CNNAdaptationModule(
                self.env.history_len, self.state_action_dim, self.env.ext_dim
            )
        raise ValueError(
            f"Unknown adaptation module architecture '{self.env.am_architecture}'."
        )

    def _get_adaptation_module_file_name(self) -> str:
        """
        Get the adaptation module checkpoint.
        """
        return f"adapt_module_state_{self.env.am_architecture}.pkl"

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
        self.adaptation_module_file_name = self._get_adaptation_module_file_name()
