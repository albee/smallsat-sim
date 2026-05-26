import os

import jax
import jax.numpy as jnp
from flax import nnx

from smallsat_sim.envs.vec_env import (
    VecEnv,
    VecEnvStepOutput,
    freeflyer_to_mjx_state,
    vecenv_step,
    vecenv_step_freeflyer,
)
from smallsat_sim.planners.base_planner import BasePlanner
from smallsat_sim.controllers.pd.vectorized_controller import VectorizedPDController
from smallsat_sim.controllers.rl.algorithms.vpg import VPG
from smallsat_sim.controllers.rl.algorithms.ppo import PPO
from smallsat_sim.controllers.rl.modules.am_cnn import CNNAdaptationModule
from smallsat_sim.controllers.rl.modules.am_transformer import (
    CrossAttentionAdaptationModule,
    TransformerAdaptationModule,
)
from smallsat_sim.controllers.rl.runners.adaptive_context import (
    authority_metrics_from_wrench,
    build_adaptation_query,
    build_adaptive_context,
    summarize_authority_metrics,
)
from smallsat_sim.controllers.rl.runners.runner_utils import load_trained_modules

_JITTED_VECENV_STEP = jax.jit(vecenv_step, static_argnames=("config",))
_JITTED_FREEFLYER_STEP = jax.jit(vecenv_step_freeflyer, static_argnames=("config",))
MAX_LOGGED_TRAJECTORY_ENVS = 10


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
        perturbation_distributions: tuple[jnp.ndarray, ...] | None = None,
        apply_disturbances: bool = False,
        perturbation_sequence: tuple[tuple[int, jnp.ndarray], ...] | None = None,
        disturbance_start_steps: tuple[int, ...] = (),
    ) -> None:
        """
        Control the agent using the previously trained RL controller.

        Adaptive deployment uses phase 2: the policy context is produced by the
        adaptation module from measured state/requested-command history.
        """
        if self.env.use_adaptive_approach and int(phase) != 2:
            raise ValueError(
                "Adaptive deployment must use phase=2. Phase 1 consumes privileged "
                "simulator force/wrench labels and is not realistic at runtime."
            )
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
        self.adaptation_module = jax.vmap(self.am, in_axes=(0, 0))

        self.env.reset()
        self.env.reset_perturbations()
        if hasattr(self.env, "reset_disturbances"):
            self.env.reset_disturbances()
        self.env._refresh_effect_states()

        rollout_backend = os.environ.get(
            "SMALLSAT_ROLLOUT_BACKEND",
            getattr(self.env.env_cfg.control.RL, "rollout_backend", "mjx"),
        )
        if rollout_backend == "freeflyer":
            vec_state = self.env.freeflyer_state_struct()
        else:
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
        applied_sequence_events: set[int] = set()
        applied_disturbance_events: set[int] = set()
        while True:
            if self.deployment_len is not None and step >= max_steps:
                break

            use_functional = not test_pd

            # Optional single-life sequence: apply multiple faults/disturbances
            # at different times without resetting the episode.
            if perturbation_sequence is not None:
                applied_event_this_step = False
                for event_idx, (start_step, distribution) in enumerate(
                    perturbation_sequence
                ):
                    if (
                        step == int(start_step)
                        and event_idx not in applied_sequence_events
                    ):
                        self.env.apply_random_perturbations(
                            key=self._take_keys(),
                            fraction_perturbed_envs=1.0,
                            perturbation_distribution=distribution,
                        )
                        applied_sequence_events.add(event_idx)
                        applied_event_this_step = True
                for event_idx, start_step in enumerate(disturbance_start_steps):
                    if (
                        step == int(start_step)
                        and event_idx not in applied_disturbance_events
                    ):
                        self.env.apply_random_disturbance(
                            key=self._take_keys(), fraction_disturbed_envs=1.0
                        )
                        applied_disturbance_events.add(event_idx)
                        applied_event_this_step = True
                if applied_event_this_step:
                    self.env._refresh_effect_states()
                    if use_functional:
                        if rollout_backend == "freeflyer":
                            vec_state = vec_state.replace(
                                disturbance_states=self.env.disturbance_states,
                                perturbation_states=self.env.perturbation_states,
                            )
                        else:
                            vec_state = self.env.state_struct

            # Standard deployment stages start one perturbation set after 100 steps.
            if (
                perturbation_sequence is None
                and
                self.deployment_len is not None
                and self.deployment_len >= 100
                and step == 100
            ):
                active_distributions = perturbation_distributions
                if active_distributions is None and perturbation_distribution is not None:
                    active_distributions = (perturbation_distribution,)
                if active_distributions is not None:
                    for distribution in active_distributions:
                        self.env.apply_random_perturbations(
                            key=self._take_keys(),
                            fraction_perturbed_envs=1.0,
                            perturbation_distribution=distribution,
                        )
                if apply_disturbances:
                    self.env.apply_random_disturbance(
                        key=self._take_keys(), fraction_disturbed_envs=1.0
                    )
                self.env._refresh_effect_states()
                if use_functional:
                    if rollout_backend == "freeflyer":
                        vec_state = vec_state.replace(
                            disturbance_states=self.env.disturbance_states,
                            perturbation_states=self.env.perturbation_states,
                        )
                    else:
                        vec_state = self.env.state_struct

            if use_functional:
                policy_input = (
                    jnp.concatenate([states, res], axis=1) if res.shape[-1] else states
                )
                critic_values = self.agent.critic.forward(policy_input)
                actions = self.agent.get_control_input(
                    stage,
                    policy_input,
                )
                step_config = self.env.build_step_config(max_episode_len=max_steps)
                if rollout_backend == "freeflyer":
                    vec_state, step_output = _JITTED_FREEFLYER_STEP(
                        vec_state, actions, next_waypoint, config=step_config
                    )
                    synced_state = freeflyer_to_mjx_state(
                        vec_state,
                        self.env.state_struct,
                        step_config,
                    )
                    self.env.apply_state_struct(synced_state)
                else:
                    vec_state, step_output = _JITTED_VECENV_STEP(
                        vec_state, actions, next_waypoint, config=step_config
                    )
                    self.env.apply_state_struct(vec_state)
            else:
                if res.shape[-1]:
                    policy_input = jnp.concatenate([states, res], axis=1)
                else:
                    policy_input = states
                critic_values = self.agent.critic.forward(policy_input)
                if test_pd:
                    prev_obs = self.env.get_obs()
                    actions = jnp.asarray(
                        self.pd_ctrl.get_control_input(self.env, next_waypoint)
                    )
                else:
                    actions = self.agent.get_control_input(stage, policy_input)
                rewards, terminals = self.env.transition(
                    actions,
                    jnp.concatenate([states, res], axis=1),
                    next_waypoint,
                )
                next_obs_batch = self.env.get_obs()
                next_states = self.env.get_states(next_waypoint)
                reward_components = self.env.get_last_reward_components()
                success_terminals = reward_components.get(
                    "terminated_success",
                    terminals.astype(rewards.dtype),
                ).astype(bool)
                failure_terminals = reward_components.get(
                    "terminated_failure",
                    jnp.zeros_like(terminals, dtype=rewards.dtype),
                ).astype(bool)
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
                    success_terminals=success_terminals,
                    failure_terminals=failure_terminals,
                    reward_components=reward_components,
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

            actual_wrench = step_output.actual_wrench
            desired_wrench = step_output.desired_wrench
            if self.env.use_adaptive_approach:
                history_full = history_counts >= history_len
                if phase == 1:
                    ext = build_adaptive_context(
                        commanded_ctrl=step_output.commanded_ctrl,
                        applied_ctrl=step_output.applied_ctrl,
                        actual_wrench=actual_wrench,
                        desired_wrench=desired_wrench,
                        previous_context=res,
                        use_adaptive_approach=True,
                        adaptive_context_mode=self.env.adaptive_context_mode,
                        thruster_mixer_T=self.env._thruster_mixer_T,
                    )
                    res = ext
                elif phase == 2:
                    query = build_adaptation_query(
                        states=step_output.prev_states,
                        desired_wrench=desired_wrench,
                        use_task_conditioned_am=self.env.use_task_conditioned_am,
                    )
                    ext_pred = self.adaptation_module(state_action_history, query)
                    ext = jnp.where(history_full[:, None], ext_pred, ext)
                    res = jnp.where(history_full[:, None], ext, res)
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
            if self.agent.has_logger:
                n_log_envs = min(self.env.num_envs, MAX_LOGGED_TRAJECTORY_ENVS)
                positions = step_output.next_obs[:n_log_envs, :3]
                pos_error = jnp.linalg.norm(step_output.next_states[:, :3], axis=1)
                att_error = jnp.linalg.norm(step_output.next_states[:, 3:6], axis=1)
                speed = jnp.linalg.norm(step_output.next_states[:, 6:9], axis=1)
                ang_speed = jnp.linalg.norm(step_output.next_states[:, 9:12], axis=1)
                in_terminal_set = jnp.logical_and(
                    pos_error <= self.env.terminal_radius,
                    jnp.logical_and(
                        speed <= self.env.terminal_max_speed,
                        jnp.logical_and(
                            att_error <= self.env.terminal_max_att_error,
                            ang_speed <= self.env.terminal_max_ang_speed,
                        ),
                    ),
                )
                authority_payload = summarize_authority_metrics(
                    authority_metrics_from_wrench(
                        commanded_ctrl=step_output.commanded_ctrl,
                        applied_ctrl=step_output.applied_ctrl,
                        desired_wrench=step_output.desired_wrench,
                        thruster_mixer_T=self.env._thruster_mixer_T,
                    )
                )
                authority_payload = {
                    key.replace("/", "_"): value
                    for key, value in authority_payload.items()
                }
                self.env.logger.log(
                    self.env.run_id,
                    float(self.env.mjx_batch.time[0]),
                    run_name=self.env.run_name,
                    stage=stage,
                    step=int(step),
                    position_x=positions[:, 0],
                    position_y=positions[:, 1],
                    position_z=positions[:, 2],
                    critic_value_mean=float(jnp.mean(critic_values)),
                    critic_value_std=float(jnp.std(critic_values)),
                    reward_mean=float(jnp.mean(step_output.rewards)),
                    terminal_set_rate=float(
                        jnp.mean(in_terminal_set.astype(jnp.float32))
                    ),
                    terminal_strict_set_rate=float(
                        jnp.mean(in_terminal_set.astype(jnp.float32))
                    ),
                    terminal_pos_ok_rate=float(
                        jnp.mean(
                            (pos_error <= self.env.terminal_radius).astype(jnp.float32)
                        )
                    ),
                    terminal_speed_ok_rate=float(
                        jnp.mean(
                            (speed <= self.env.terminal_max_speed).astype(jnp.float32)
                        )
                    ),
                    terminal_att_ok_rate=float(
                        jnp.mean(
                            (att_error <= self.env.terminal_max_att_error).astype(
                                jnp.float32
                            )
                        )
                    ),
                    terminal_ang_speed_ok_rate=float(
                        jnp.mean(
                            (ang_speed <= self.env.terminal_max_ang_speed).astype(
                                jnp.float32
                            )
                        )
                    ),
                    terminal_set_pos_error_mean=float(jnp.mean(pos_error)),
                    terminal_set_speed_mean=float(jnp.mean(speed)),
                    terminal_rate=float(
                        jnp.mean(step_output.terminals.astype(jnp.float32))
                    ),
                    success_terminal_rate=float(
                        jnp.mean(step_output.success_terminals.astype(jnp.float32))
                    ),
                    failure_terminal_rate=float(
                        jnp.mean(step_output.failure_terminals.astype(jnp.float32))
                    ),
                    **authority_payload,
                )

            if self.planner.completed_path.all():
                break

            step += 1

    def _build_adaptation_module(self):
        """
        Build the right adaptation module.
        """
        rngs = nnx.Rngs(params=self._take_keys(), dropout=self._take_keys())
        predict_delta_dim = (
            self.env.obs_dim
            if float(self.env.env_cfg.control.RL.am_predict_delta_weight) > 0.0
            else 0
        )
        predict_tracking = (
            float(self.env.env_cfg.control.RL.am_predict_tracking_weight) > 0.0
        )
        if self.env.am_architecture == "transformer":
            return TransformerAdaptationModule(
                self.env.history_len,
                self.state_action_dim,
                self.env.ext_dim,
                query_dim=self.env.am_query_dim,
                predict_delta_dim=predict_delta_dim,
                predict_tracking=predict_tracking,
                rngs=rngs,
            )
        if self.env.am_architecture == "transformer_cross_attention":
            return CrossAttentionAdaptationModule(
                self.env.history_len,
                self.state_action_dim,
                self.env.ext_dim,
                query_dim=self.env.am_query_dim,
                predict_delta_dim=predict_delta_dim,
                predict_tracking=predict_tracking,
                rngs=rngs,
            )
        if self.env.am_architecture == "cnn":
            return CNNAdaptationModule(
                self.env.history_len,
                self.state_action_dim,
                self.env.ext_dim,
                query_dim=self.env.am_query_dim,
                predict_delta_dim=predict_delta_dim,
                predict_tracking=predict_tracking,
                rngs=rngs,
            )
        raise ValueError(
            f"Unknown adaptation module architecture '{self.env.am_architecture}'."
        )

    def _get_adaptation_module_file_name(self) -> str:
        """
        Get the adaptation module checkpoint.
        """
        parts = [f"adapt_module_state_{self.env.am_architecture}"]
        if self.env.use_adaptive_approach:
            parts.append("adaptive")
            parts.append(self.env.adaptive_context_mode)
        predictive_am = (
            self.env.use_task_conditioned_am
            or float(self.env.env_cfg.control.RL.am_predict_delta_weight) > 0.0
            or float(self.env.env_cfg.control.RL.am_predict_tracking_weight) > 0.0
        )
        if predictive_am:
            parts.append("taskpred")
        if self.env.use_pretrained:
            parts.append("pretrained")
        if not self.env.train_with_failures:
            parts.append("nominal")
        return "_".join(parts) + ".pkl"

    def _get_training_state_file_name(self) -> None:
        """
        Set the checkpoint file name for the training state.
        """
        parts = ["training_state"]

        if self.env.use_adaptive_approach:
            parts.append("adaptive")
            parts.append(self.env.adaptive_context_mode)
        if self.env.use_pretrained:
            parts.append("pretrained")
        if not self.env.train_with_failures:
            parts.append("nominal")

        self.ckpt_filename = "_".join(parts) + ".pkl"
        self.adaptation_module_file_name = self._get_adaptation_module_file_name()
