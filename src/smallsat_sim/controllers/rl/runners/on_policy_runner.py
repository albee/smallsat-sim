import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import wandb

from smallsat_sim.envs.vec_env import (
    VecEnv,
    VecEnvState,
    VecEnvStepConfig,
    VecEnvStepOutput,
    vecenv_step,
    vecenv_reset_to_config,
    _compute_state_features,
)
from smallsat_sim.planners.oracle.oracle_rl import OraclePlannerRL
from smallsat_sim.controllers.pd.vectorized_controller import VectorizedPDController
from smallsat_sim.controllers.rl.algorithms.vpg import VPG
from smallsat_sim.controllers.rl.algorithms.ppo import PPO
from smallsat_sim.controllers.rl.modules.am_cnn import CNNAdaptationModule
from smallsat_sim.controllers.rl.modules.am_transformer import (
    TransformerAdaptationModule,
)
from smallsat_sim.controllers.rl.storage.replay_buffer import ReplayBuffer
from smallsat_sim.controllers.rl.runners.runner_utils import (
    save_training_data,
    save_trained_modules,
    save_adaptation_module,
    load_training_data,
    load_trained_modules,
)
from smallsat_sim.utils.helpers_jax import (
    train_val_split,
    mae_loss_fn,
    calc_lateral_tracking_error,
    calc_attitude_error,
    calc_extrinsic_error,
)
from smallsat_sim.utils.wandb_config import setup_wandb


@dataclass
class FunctionalRolloutCallbacks:
    """
    Collection of callables used by `run_functional_rollout` to interact with
    policy logic outside the environment stepping loop.

    Each callable receives the current step index together with the evolving
    carry so callers can maintain additional per-rollout state (e.g. history
    buffers for the adaptation module).
    """

    prepare_policy_input: Callable[
        [int, jnp.ndarray, jnp.ndarray, Any], tuple[jnp.ndarray, Any]
    ]
    sample_policy: Callable[
        [int, jnp.ndarray, jnp.ndarray, Any],
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Any],
    ]
    post_step: Callable[
        [int, VecEnvStepOutput, jnp.ndarray, jnp.ndarray, bool, Any],
        tuple[jnp.ndarray, Any, Any],
    ]
    bootstrap_value: Callable[
        [int, VecEnvState, jnp.ndarray, jnp.ndarray, Any],
        tuple[jnp.ndarray, jnp.ndarray, Any],
    ]


@dataclass
class FunctionalRolloutResult:
    """
    Batched outputs produced by `run_functional_rollout`.
    """

    step_outputs: VecEnvStepOutput
    actions: jnp.ndarray
    values: jnp.ndarray
    logp: jnp.ndarray
    residuals: jnp.ndarray
    episode_returns: jnp.ndarray
    done_flags: jnp.ndarray
    bootstrap_values: jnp.ndarray
    aux: Any
    final_state: VecEnvState
    final_residuals: jnp.ndarray
    final_rng: jnp.ndarray
    final_extra: Any


@dataclass
class AdaptationRolloutExtra:
    history: jnp.ndarray
    counts: jnp.ndarray


def _adaptation_rollout_extra_flatten(extra: "AdaptationRolloutExtra"):
    children = (extra.history, extra.counts)
    return children, None


def _adaptation_rollout_extra_unflatten(aux_data, children):
    history, counts = children
    return AdaptationRolloutExtra(history=history, counts=counts)


jax.tree_util.register_pytree_node(
    AdaptationRolloutExtra,
    _adaptation_rollout_extra_flatten,
    _adaptation_rollout_extra_unflatten,
)


@dataclass
class _FunctionalRolloutStep:
    step_output: VecEnvStepOutput
    actions: jnp.ndarray
    values: jnp.ndarray
    logp: jnp.ndarray
    residuals: jnp.ndarray
    episode_return: jnp.ndarray
    done_flag: jnp.ndarray
    bootstrap_value: jnp.ndarray
    aux: Any


def _functional_rollout_step_flatten(step: "_FunctionalRolloutStep"):
    children = (
        step.step_output,
        step.actions,
        step.values,
        step.logp,
        step.residuals,
        step.episode_return,
        step.done_flag,
        step.bootstrap_value,
        step.aux,
    )
    return children, None


def _functional_rollout_step_unflatten(aux_data, children):
    (
        step_output,
        actions,
        values,
        logp,
        residuals,
        episode_return,
        done_flag,
        bootstrap_value,
        aux,
    ) = children
    return _FunctionalRolloutStep(
        step_output=step_output,
        actions=actions,
        values=values,
        logp=logp,
        residuals=residuals,
        episode_return=episode_return,
        done_flag=done_flag,
        bootstrap_value=bootstrap_value,
        aux=aux,
    )


jax.tree_util.register_pytree_node(
    _FunctionalRolloutStep,
    _functional_rollout_step_flatten,
    _functional_rollout_step_unflatten,
)


def run_functional_rollout(
    *,
    step_config: VecEnvStepConfig,
    initial_state: VecEnvState,
    initial_residuals: jnp.ndarray,
    rng: jnp.ndarray,
    num_steps: int,
    reference_waypoint: jnp.ndarray,
    callbacks: FunctionalRolloutCallbacks,
    extra: Any = None,
) -> FunctionalRolloutResult:
    """
    Advance the vectorised environment purely functionally for ``num_steps``.

    Invariants:
    - The helper mirrors the imperative `VecEnv.transition` semantics, including
      the `control_decimation` inner loop and per-epoch resets when all
      environments terminate or the configured horizon is reached.
    - `initial_residuals` must have shape `(num_envs, res_dim)` (or `(num_envs, 0)`
      when residuals are disabled).
    - Callback implementations must be side-effect free; any mutable state should
      be threaded via the `extra` carry.
    - The returned `step_outputs` contain the per-step state snapshots needed to
      reconstruct rewards, metrics, and logging payloads without reaching back
      into the imperative environment.
    """

    num_envs = initial_state.mjx_batch.qpos.shape[0]

    def _initial_episode_state():
        return (
            initial_state,
            initial_residuals,
            rng,
            extra,
            jnp.zeros((num_envs,), dtype=jnp.float32),
            jnp.array(0, dtype=jnp.int32),
        )

    def _prepare_policy_input(
        step_idx: int, states: jnp.ndarray, residuals: jnp.ndarray, carry_extra: Any
    ):
        return callbacks.prepare_policy_input(step_idx, states, residuals, carry_extra)

    def _sample_policy(
        step_idx: int,
        policy_input: jnp.ndarray,
        rng_key: jnp.ndarray,
        carry_extra: Any,
    ):
        return callbacks.sample_policy(step_idx, policy_input, rng_key, carry_extra)

    def _post_step(
        step_idx: int,
        step_output: VecEnvStepOutput,
        actions: jnp.ndarray,
        residuals: jnp.ndarray,
        reset_pending: bool,
        carry_extra: Any,
    ):
        return callbacks.post_step(
            step_idx, step_output, actions, residuals, reset_pending, carry_extra
        )

    def _bootstrap_value(
        step_idx: int,
        env_state: VecEnvState,
        residuals: jnp.ndarray,
        rng_key: jnp.ndarray,
        carry_extra: Any,
    ):
        return callbacks.bootstrap_value(
            step_idx, env_state, residuals, rng_key, carry_extra
        )

    def _scan_body(
        carry: tuple[
            VecEnvState, jnp.ndarray, jnp.ndarray, Any, jnp.ndarray, jnp.ndarray
        ],
        step_idx: int,
    ):
        env_state, residuals, rng_key, carry_extra, ep_ret, ep_len = carry

        states_curr = _compute_state_features(env_state.mjx_batch, reference_waypoint)
        policy_input, carry_extra = _prepare_policy_input(
            step_idx, states_curr, residuals, carry_extra
        )
        actions, values, logp, rng_key, carry_extra = _sample_policy(
            step_idx, policy_input, rng_key, carry_extra
        )

        next_env_state, step_output = vecenv_step(
            env_state,
            actions,
            reference_waypoint,
            step_config,
            residuals,
        )
        def _log_nan(_):
            jax.debug.print("NaN in next_obs at step {s}", s=step_idx)
            return jnp.array(0, dtype=jnp.int32)

        _ = jax.lax.cond(
            jnp.isnan(step_output.next_obs).any(),
            _log_nan,
            lambda _: jnp.array(0, dtype=jnp.int32),
            operand=None,
        )

        ep_ret_next = ep_ret + step_output.rewards
        ep_len_next = ep_len + 1

        all_terminal = jnp.all(step_output.terminals)
        timeout = ep_len_next >= step_config.max_episode_len
        epoch_last = jnp.equal(step_idx, num_steps - 1)
        done_without_epoch = jnp.logical_or(all_terminal, timeout)
        done_flag = jnp.logical_or(done_without_epoch, epoch_last)

        next_residuals, step_aux, carry_extra = _post_step(
            step_idx, step_output, actions, residuals, done_without_epoch, carry_extra
        )

        bootstrap_values, rng_key, carry_extra = jax.lax.cond(
            epoch_last,
            lambda args: _bootstrap_value(step_idx, *args),
            lambda args: (jnp.zeros_like(step_output.rewards), args[2], args[3]),
            operand=(next_env_state, next_residuals, rng_key, carry_extra),
        )

        episode_return = jax.lax.cond(
            done_flag,
            lambda _: ep_ret_next,
            lambda _: jnp.zeros_like(ep_ret_next),
            operand=None,
        )

        def _reset_after_done(_):
            reset_state = vecenv_reset_to_config(next_env_state, step_config)
            zero_residuals = jnp.zeros_like(initial_residuals)
            zero_return = jnp.zeros((num_envs,), dtype=ep_ret_next.dtype)
            zero_length = jnp.array(0, dtype=ep_len_next.dtype)
            return reset_state, zero_residuals, zero_return, zero_length

        next_env_state, next_residuals, ep_ret_final, ep_len_final = jax.lax.cond(
            jnp.logical_and(done_without_epoch, jnp.logical_not(epoch_last)),
            _reset_after_done,
            lambda _: (next_env_state, next_residuals, ep_ret_next, ep_len_next),
            operand=None,
        )

        step_record = _FunctionalRolloutStep(
            step_output=step_output,
            actions=actions,
            values=values,
            logp=logp,
            residuals=next_residuals,
            episode_return=episode_return,
            done_flag=done_flag,
            bootstrap_value=bootstrap_values,
            aux=step_aux,
        )

        new_carry = (
            next_env_state,
            next_residuals,
            rng_key,
            carry_extra,
            ep_ret_final,
            ep_len_final,
        )
        return new_carry, step_record

    initial_carry = _initial_episode_state()
    (final_state, final_residuals, final_rng, final_extra, _, _), steps = jax.lax.scan(
        _scan_body,
        initial_carry,
        jnp.arange(num_steps, dtype=jnp.int32),
    )

    step_outputs = steps.step_output
    actions = steps.actions
    values = steps.values
    logp = steps.logp
    residuals = steps.residuals
    episode_returns = steps.episode_return
    done_flags = steps.done_flag
    bootstrap_values = steps.bootstrap_value
    aux = steps.aux

    return FunctionalRolloutResult(
        step_outputs=step_outputs,
        actions=actions,
        values=values,
        logp=logp,
        residuals=residuals,
        episode_returns=episode_returns,
        done_flags=done_flags,
        bootstrap_values=bootstrap_values,
        aux=aux,
        final_state=final_state,
        final_residuals=final_residuals,
        final_rng=final_rng,
        final_extra=final_extra,
    )


class OnPolicyRunner(object):
    """
    On-policy runner for training and evaluation. Inspired from https://spinningup.openai.com/en/latest/algorithms/vpg.html.
    """

    def __init__(self, env: VecEnv, planner: OraclePlannerRL) -> None:
        # Initialize the environment and agent
        self.env = env
        self.planner = planner
        base_key = self.env.next_rng_keys(1)[0]
        self._rng, agent_key = jax.random.split(base_key)
        self.agent = PPO(self.env, planner, rng_key=agent_key)
        self.state_action_dim = self.env.obs_dim + self.env.act_dim
        self.am = self._build_adaptation_module()
        self.reference_point = planner.get_reference(
            self.env.get_obs()
        )  # Is re-used in every episode
        self.pd_ctrl = VectorizedPDController(env, planner)
        self._load_rl_hyperparams()

        # Vectorize adaptation module
        self.adaptation_module = jax.vmap(self.am)
        self.am_kl_weight = self.env.env_cfg.control.RL.am_kl_weight
        self.am_loss_fn = self._select_am_loss_fn()

        # JIT-compile the adaptation module updates
        self.jitted_batched_am_loss_and_grad = nnx.jit(
            nnx.value_and_grad(self.am_loss_fn), static_argnums=()
        )

        rl_cfg = self.env.env_cfg.control.RL
        self._functional_check_enabled = bool(
            getattr(rl_cfg, "verify_functional_rollout", False)
        )
        self._functional_check_ran = False
        self._functional_check_atol = float(
            getattr(rl_cfg, "verify_functional_rollout_atol", 1e-4)
        )
        self._functional_check_rtol = float(
            getattr(rl_cfg, "verify_functional_rollout_rtol", 1e-3)
        )

        # Path to save the checkpoints
        self.ckpt_dir = "src/smallsat_sim/controllers/rl/checkpoints/"

        # Checkpoint file names
        self._create_checkpoint_file_names()

        # Use Weights and Biases for logging
        if self.env.use_wandb:
            setup_wandb()
            wandb.init(
                project="Astrobee-training",
                name=f"{self.env.run_name}_{self.env.run_id}",
                config={
                    "num_envs": self.env.num_envs,
                    "steps_per_epoch": self.steps_per_epoch,
                    "epochs": self.epochs,
                    "max_ep_len": self.max_ep_len,
                    "gamma": self.gamma,
                    "lam": self.lam,
                    "actor_lr": self.agent.actor_lr,
                    "critic_lr": self.agent.critic_lr,
                    "episode_len": self.episode_len,
                    "n_evals": self.n_evals,
                },
            )

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

    def pretrain(self, strategy: str = "supervised_learning") -> None:
        """
        Pretrain the actor and critic networks.
        """
        # Check if pretraining has already been done
        file_path = os.path.join(self.ckpt_dir, self.pretraining_state_file_name)
        if os.path.isfile(file_path):
            return

        print("Pretraining modules...\n")

        # Generate experience if necessary and load the pretraining data
        self._generate_experience()
        pretraining_data = load_training_data(
            self.ckpt_dir, self.pretraining_data_file_name
        )

        # Load the data
        obs = pretraining_data["obs"].reshape(-1, self.env.obs_dim)
        act = pretraining_data["act"].reshape(-1, self.env.act_dim)
        ret = pretraining_data["ret"].reshape(-1)
        tdres = pretraining_data["tdres"].reshape(-1)
        logp = pretraining_data["logp"].reshape(-1)
        if self.env.use_adaptive_approach is True:
            residuals = pretraining_data["residuals"].reshape(-1, self.env.res_dim)
        else:
            residuals = jnp.empty((self.steps_per_epoch * self.env.num_envs, 0))

        # Clip the actions to the highest upper bound on the force range of the thrusters
        act_clipped = jnp.where(act > 0.6, 0.6, act)

        # Pretrain the policy network
        if strategy == "supervised_learning":
            # Optimizer to pretrain the policy network
            actor_optimizer = nnx.Optimizer(
                self.agent.actor, optax.adam(learning_rate=1e-2, eps=1e-5)
            )

            # Split into training and validation sets
            split_key = self._take_keys()
            (
                X_train,
                y_train,
                X_val,
                y_val,
                _,
            ) = train_val_split(
                jnp.concatenate([obs, residuals], axis=1),
                act_clipped,
                key=split_key,
            )

            num_epochs = 80
            num_batches = 256
            batch_size = int(jnp.ceil(obs.shape[0] / num_batches))
            num_train_samples = X_train.shape[0]
            num_val_samples = X_val.shape[0]

            # Create PRNG keys
            subkeys_pretrain = self._take_keys(num_epochs)

            # Training loop
            for epoch in range(num_epochs):
                # Shuffle the training data
                indices = jax.random.permutation(
                    subkeys_pretrain[epoch], num_train_samples
                )
                X_train = X_train[indices]
                y_train = y_train[indices]

                epoch_train_loss_sum = jnp.array(0.0)
                num_train_batches = 0
                for i in range(0, num_train_samples, batch_size):
                    batch_X = X_train[i : i + batch_size]
                    batch_y = y_train[i : i + batch_size]

                    # Train network
                    actor_loss, grads = nnx.value_and_grad(mae_loss_fn)(
                        self.agent.actor, batch_X, batch_y, subkeys_pretrain[epoch]
                    )
                    actor_optimizer.update(grads)
                    epoch_train_loss_sum = epoch_train_loss_sum + actor_loss
                    num_train_batches += 1

                avg_train_loss = epoch_train_loss_sum / max(num_train_batches, 1)
                print(
                    f"Epoch: {epoch+1:2} avg. actor training loss: {avg_train_loss}\n"
                )

                epoch_val_loss_sum = jnp.array(0.0)
                num_val_batches = 0
                for i in range(0, num_val_samples, batch_size):
                    batch_X_val = X_val[i : i + batch_size]
                    batch_y_val = y_val[i : i + batch_size]

                    # Validate
                    actor_val_loss, _ = nnx.value_and_grad(mae_loss_fn)(
                        self.agent.actor,
                        batch_X_val,
                        batch_y_val,
                        subkeys_pretrain[epoch],
                    )
                    epoch_val_loss_sum = epoch_val_loss_sum + actor_val_loss
                    num_val_batches += 1

                avg_val_loss = epoch_val_loss_sum / max(num_val_batches, 1)
                print(
                    f"Epoch: {epoch+1:2} avg. actor evaluation loss: {avg_val_loss}\n"
                )
                if self.env.use_wandb:
                    wandb.log(
                        {
                            "training_loss": float(avg_train_loss),
                            "validation_loss": float(avg_val_loss),
                        }
                    )
        elif strategy == "rl":
            self.agent.update_policy_gradient(
                self._take_keys(), obs, act_clipped, tdres, logp
            )
        else:
            raise Exception(
                "This strategy does not exist. Options are [supervised_learning] and [rl]."
            )

        # Pretrain the base network
        self.agent.update_value_function(
            self._take_keys(),
            jnp.concatenate([obs, residuals], axis=1),
            ret,
        )

        # Only pretrain the actor mean, reset log_std
        self.agent.actor.log_std = nnx.Param(-0.5 * jnp.ones(self.env.act_dim))

        # Save the trained actor and critic network weights
        save_trained_modules(
            self.agent, self.ckpt_dir, self.pretraining_state_file_name
        )

    def learn(self) -> None:
        """
        Main training loop.
        """
        # Check if training has already been done
        file_path = os.path.join(self.ckpt_dir, self.training_state_file_name)
        if os.path.isfile(file_path):
            return

        if self.env.use_pretrained:
            # Check if pretrained actor and critic modules are available and load them
            file_path = os.path.join(self.ckpt_dir, self.pretraining_state_file_name)
            if os.path.isfile(file_path):
                restored_state = load_trained_modules(
                    self.ckpt_dir, self.pretraining_state_file_name
                )
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
            else:
                print("No pretrained modules available.\n")

        print("Training agent...\n")

        # Set up buffer
        buffer = ReplayBuffer(
            self.env.num_envs,
            self.env.obs_dim,
            self.env.act_dim,
            self.env.res_dim,
            self.steps_per_epoch,
            self.gamma,
            self.lam,
        )

        # Trigger compilation of jitted updates before the main loop
        self.agent.warmup_jit()

        # Initialize the environment
        self.env.reset()
        self.env.reset_perturbations()
        episode_counter = 0

        # Main training loop
        for epoch in range(self.epochs):
            epoch_start_time = time.perf_counter()
            epoch_key = self._take_keys()
            actor_key = epoch_key
            # Apply perturbations ramp-up
            if self.env.train_with_failures and epoch >= self.epochs // 2:
                ramp_duration = max(self.epochs // 2, 1)
                ramp_progress = min((epoch - ramp_duration) / ramp_duration, 1.0)
                self.env.reset_perturbations()  # avoid accumulating failures across epochs
                perturb_key, disturb_key, actor_key = jax.random.split(epoch_key, 3)
                self.env.apply_random_perturbations(
                    key=perturb_key,
                    fraction_perturbed_envs=0.05
                    + (0.5 - 0.05) * max(ramp_progress, 0.0),
                )
                self.env.apply_random_disturbance(
                    key=disturb_key,
                    fraction_disturbed_envs=0.05
                    + (0.15 - 0.05) * max(ramp_progress, 0.0),
                )

            # Accumulate rollout stats to emit once per epoch
            rollout_timer_start = time.perf_counter()
            scan_start = rollout_timer_start
            step_config = self.env.build_step_config(max_episode_len=self.max_ep_len)
            initial_state = self.env.state_struct
            actor_state, critic_state = self.agent.actor_critic_state()

            if self.env.use_adaptive_approach:
                residual_init = jnp.zeros(
                    (self.env.num_envs, self.env.res_dim), dtype=jnp.float32
                )
            else:
                residual_init = jnp.zeros((self.env.num_envs, 0), dtype=jnp.float32)

            def _prepare_policy_input(_step, states, residuals, carry_extra):
                del _step  # unused
                if residuals.shape[-1]:
                    return jnp.concatenate([states, residuals], axis=1), carry_extra
                return states, carry_extra

            def _sample_policy(_step, policy_input, rng_key, carry_extra):
                del _step  # unused
                rng_key, sample_key = jax.random.split(rng_key)
                actions, values, logp = self.agent.functional_act(
                    actor_state,
                    critic_state,
                    policy_input,
                    sample_key,
                )
                return actions, values, logp, rng_key, carry_extra

            def _post_step(
                _step, step_output, actions, residuals, reset_flag, carry_extra
            ):
                del _step, actions, reset_flag  # unused
                if self.env.use_adaptive_approach:
                    residuals_next = (
                        step_output.actual_wrench - step_output.desired_wrench
                    )
                else:
                    residuals_next = residuals
                return residuals_next, None, carry_extra

            def _bootstrap_value(step_idx, env_state, residuals, rng_key, carry_extra):
                rng_key, value_key = jax.random.split(rng_key)
                next_states = _compute_state_features(
                    env_state.mjx_batch, self.reference_point
                )
                policy_input, carry_extra = _prepare_policy_input(
                    step_idx, next_states, residuals, carry_extra
                )
                _, values, _ = self.agent.functional_act(
                    actor_state,
                    critic_state,
                    policy_input,
                    value_key,
                )
                return values, rng_key, carry_extra

            rollout_result = run_functional_rollout(
                step_config=step_config,
                initial_state=initial_state,
                initial_residuals=residual_init,
                rng=self.agent.key,
                num_steps=self.steps_per_epoch,
                reference_waypoint=self.reference_point,
                callbacks=FunctionalRolloutCallbacks(
                    prepare_policy_input=_prepare_policy_input,
                    sample_policy=_sample_policy,
                    post_step=_post_step,
                    bootstrap_value=_bootstrap_value,
                ),
            )

            jax.block_until_ready(rollout_result.actions)
            scan_time = time.perf_counter() - scan_start
            self.agent.key = rollout_result.final_rng

            if (
                self._functional_check_enabled
                and not self._functional_check_ran
                and rollout_result.actions.shape[0] > 0
            ):
                self.env.verify_functional_step(
                    initial_state,
                    rollout_result.actions[0],
                    self.reference_point,
                    step_config=step_config,
                    atol=self._functional_check_atol,
                    rtol=self._functional_check_rtol,
                )
                self._functional_check_ran = True

            self.env.apply_state_struct(rollout_result.final_state)

            step_outputs = rollout_result.step_outputs
            actions_traj = rollout_result.actions
            values_traj = rollout_result.values
            logp_traj = rollout_result.logp
            residuals_traj = rollout_result.residuals
            done_flags = rollout_result.done_flags
            bootstrap_vals = rollout_result.bootstrap_values
            episode_returns_traj = rollout_result.episode_returns

            terminal_count = float(jnp.sum(step_outputs.terminals.astype(jnp.float32)))

            buffer_start = time.perf_counter()
            start_ptr = buffer.path_start_idx
            buffer.store_batch(
                step_outputs.prev_states,
                actions_traj,
                step_outputs.rewards,
                values_traj,
                logp_traj,
                residuals_traj,
            )

            done_indices = jnp.nonzero(
                done_flags, size=self.steps_per_epoch, fill_value=-1
            )[0]
            valid_done = done_indices[done_indices >= 0]
            episode_return_sum = 0.0
            episode_counter_epoch = 0

            if valid_done.size > 0:
                end_ptrs = start_ptr + valid_done + 1
                buffer.end_traj_batch(end_ptrs.tolist(), bootstrap_vals[valid_done])

                mean_returns = episode_returns_traj[valid_done].mean(axis=1)
                episode_return_sum = float(jnp.sum(mean_returns))
                episode_counter_epoch = int(mean_returns.shape[0])

                if self.agent.has_logger:
                    for mean_value in mean_returns.tolist():
                        self.env.logger.log(
                            self.env.run_id,
                            float(self.env.mjx_batch.time[0]),
                            step=episode_counter,
                            run_name=self.env.run_name,
                            stage="policy_training",
                            mean_episodic_returns=mean_value,
                        )
                        episode_counter += 1

            buffer_time = time.perf_counter() - buffer_start

            epoch_reward_components = (
                step_outputs.reward_components
                if self.env.collect_reward_components
                else {}
            )

            self.env.reset()
            self.env.reset_perturbations()

            rollout_duration = time.perf_counter() - epoch_start_time

            # Get the data from the training loop and save it
            data = buffer.get()
            if epoch == self.epochs - 1:
                save_training_data(self.ckpt_dir, self.training_data_file_name, data)

            obs = data["obs"].reshape(-1, self.env.obs_dim)
            actions = data["act"].reshape(-1, self.env.act_dim)
            rews = data["rews"].reshape(-1)
            tdres = data["tdres"].reshape(-1)
            returns = data["ret"].reshape(-1)
            logp = data["logp"].reshape(-1)
            if self.env.use_adaptive_approach is True:
                residuals = data["residuals"].reshape(-1, self.env.res_dim)
            else:
                residuals = jnp.empty((self.steps_per_epoch * self.env.num_envs, 0))

            if obs.size:
                tracking_error_epoch = calc_lateral_tracking_error(
                    obs, self.planner
                ).mean()
                angle_error_epoch = jnp.degrees(calc_attitude_error(obs)).mean()
            else:
                tracking_error_epoch = jnp.array(0.0)
                angle_error_epoch = jnp.array(0.0)

            terminal_count_epoch = jnp.asarray(terminal_count)
            mean_ep_return_epoch = (
                jnp.asarray(episode_return_sum / episode_counter_epoch)
                if episode_counter_epoch > 0
                else jnp.array(0.0)
            )

            reward_component_means = {
                name: values.mean() for name, values in epoch_reward_components.items()
            }
            other_rollout_time = max(0.0, rollout_duration - (scan_time + buffer_time))

            update_start_time = time.perf_counter()
            # # Policy gradient update
            # actor_loss = self.agent.update_policy_gradient(
            #     subkeys_train[epoch],
            #     jnp.concatenate([obs, residuals], axis=1),
            #     actions,
            #     tdres,
            #     logp,
            # )

            # # Value function updates
            # critic_loss = self.agent.update_value_function(
            #     subkeys_train[epoch],
            #     jnp.concatenate([obs, residuals], axis=1),
            #     returns,
            # )

            # Update the policy gradient and the value function
            (
                last_actor_loss,
                last_critic_loss,
                mean_actor_loss,
                mean_critic_loss,
            ) = self.agent.update_actor_critic_minibatch(
                actor_key,
                jnp.concatenate([obs, residuals], axis=1),
                actions,
                tdres,
                logp,
                returns,
            )
            update_duration = time.perf_counter() - update_start_time
            epoch_total_duration = time.perf_counter() - epoch_start_time
            print(
                f"[Timing] Epoch {epoch + 1}/{self.epochs}: "
                f"rollout {rollout_duration:.2f}s "
                f"(scan {scan_time:.2f}s, buffer {buffer_time:.2f}s, other {other_rollout_time:.2f}s) | "
                f"update {update_duration:.2f}s | total {epoch_total_duration:.2f}s"
            )

            actor_loss_last_f = float(last_actor_loss)
            critic_loss_last_f = float(last_critic_loss)
            actor_loss_mean_f = float(mean_actor_loss)
            critic_loss_mean_f = float(mean_critic_loss)

            # Monitor key RL metrics during training using Weights & Biases
            if self.env.use_wandb:
                wandb_reward_payload = {
                    f"reward_components/{metric_name}": float(metric_value)
                    for metric_name, metric_value in reward_component_means.items()
                }
                wandb.log(
                    {
                        "mean_rewards": float(rews.mean()),
                        "actor_loss_last": actor_loss_last_f,
                        "critic_loss_last": critic_loss_last_f,
                        "actor_loss_mean": actor_loss_mean_f,
                        "critic_loss_mean": critic_loss_mean_f,
                        "mean_episodic_returns": float(mean_ep_return_epoch),
                        "num_terminal": float(terminal_count_epoch),
                        "mean_log_std": float(self.agent.actor.log_std.value.mean()),
                        "mean_std": float(
                            jnp.exp(self.agent.actor.log_std.value).mean()
                        ),
                        "mean_tracking_error": float(tracking_error_epoch),
                        "mean_angle_error": float(angle_error_epoch),
                        **wandb_reward_payload,
                    },
                )

            # Log key RL metrics
            if self.agent.has_logger:
                self.env.logger.log(
                    self.env.run_id,
                    float(self.env.mjx_batch.time[0]),
                    step=int(epoch),
                    run_name=self.env.run_name,
                    stage="policy_training",
                    mean_rewards=float(rews.mean()),
                    actor_loss_last=actor_loss_last_f,
                    critic_loss_last=critic_loss_last_f,
                    actor_loss_mean=actor_loss_mean_f,
                    critic_loss_mean=critic_loss_mean_f,
                    num_terminal=float(terminal_count_epoch),
                    mean_log_std=float(self.agent.actor.log_std.value.mean()),
                    mean_std=float(jnp.exp(self.agent.actor.log_std.value).mean()),
                    mean_tracking_error=float(tracking_error_epoch),
                    mean_angle_error=float(angle_error_epoch),
                )

            # Save the trained actor and critic network weights
            save_trained_modules(
                self.agent, self.ckpt_dir, self.training_state_file_name
            )

    def train_adaptation_module_on_policy(self) -> None:
        """
        Train adaptation module to predict extrinsics from the history of states and actions with
        on-policy data (RMA approach).
        NOTE: use_adaptive_approach must be set to True in the environment config.
        """
        if self.env.use_adaptive_approach is False:
            return

        file_path = os.path.join(self.ckpt_dir, self.adaptation_module_file_name)
        if os.path.isfile(file_path):
            return

        file_path = os.path.join(self.ckpt_dir, self.training_state_file_name)
        if os.path.isfile(file_path):
            restored_state = load_trained_modules(
                self.ckpt_dir, self.training_state_file_name
            )
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
        else:
            raise Exception(
                "The base policy must be trained before the adaptation module."
            )

        print("Training adaptation module...\n")

        self.env.reset()
        self.env.reset_perturbations()

        subkeys_train = self._take_keys(self.epochs)

        am_lr = self.env.env_cfg.control.RL.am_lr
        am_weight_decay = self.env.env_cfg.control.RL.am_weight_decay
        grad_clip_norm = max(float(self.env.env_cfg.control.RL.am_grad_clip_norm), 1e-6)
        am_optax = optax.chain(
            optax.clip_by_global_norm(grad_clip_norm),
            optax.adamw(
                learning_rate=am_lr,
                eps=1e-8,
                weight_decay=am_weight_decay,
            ),
        )
        self.am_optimizer = nnx.Optimizer(self.am, am_optax)

        num_envs = self.env.num_envs
        history_len = self.env.history_len
        state_action_dim = self.state_action_dim

        for epoch in range(self.epochs):
            ramp_duration = max(self.epochs // 2, 1)
            ramp_progress = min((epoch - ramp_duration) / ramp_duration, 1.0)
            self.env.reset_perturbations()
            self.env.apply_random_perturbations(
                key=subkeys_train[epoch],
                fraction_perturbed_envs=0.05 + (0.5 - 0.05) * max(ramp_progress, 0.0),
            )
            self.env.apply_random_disturbance(
                key=subkeys_train[epoch],
                fraction_disturbed_envs=0.05 + (0.15 - 0.05) * max(ramp_progress, 0.0),
            )

            step_config = self.env.build_step_config(max_episode_len=self.max_ep_len)
            vec_state = self.env.state_struct
            actor_state, critic_state = self.agent.actor_critic_state()

            def _prepare_policy_input(_step, states, residuals, carry_extra):
                del _step  # unused
                if residuals.shape[-1]:
                    return jnp.concatenate([states, residuals], axis=1), carry_extra
                return states, carry_extra

            def _sample_policy(_step, policy_input, rng_key, carry_extra):
                del _step  # unused
                actions = self.agent.get_control_input("am_training", policy_input)
                values = jnp.zeros((num_envs,), dtype=jnp.float32)
                logp = jnp.zeros((num_envs,), dtype=jnp.float32)
                return actions, values, logp, rng_key, carry_extra

            def _post_step(
                _step, step_output, actions, residuals, reset_flag, carry_extra
            ):
                del _step, residuals  # unused
                history = carry_extra.history
                counts = carry_extra.counts

                combined = jnp.concatenate([step_output.prev_states, actions], axis=1)
                history = jnp.roll(history, shift=-1, axis=1)
                history = history.at[:, -1, :].set(combined)
                counts = jnp.minimum(counts + 1, history_len)

                def _reset_hist(_):
                    return (
                        jnp.zeros_like(history),
                        jnp.zeros_like(counts),
                    )

                history, counts = jax.lax.cond(
                    reset_flag,
                    _reset_hist,
                    lambda _: (history, counts),
                    operand=None,
                )

                history_full = counts >= history_len
                residuals_next = step_output.actual_wrench - step_output.desired_wrench
                new_extra = AdaptationRolloutExtra(history=history, counts=counts)
                return residuals_next, history_full, new_extra

            def _bootstrap_value(step_idx, env_state, residuals, rng_key, carry_extra):
                rng_key, value_key = jax.random.split(rng_key)
                next_states = _compute_state_features(
                    env_state.mjx_batch, self.reference_point
                )
                policy_input, carry_extra = _prepare_policy_input(
                    step_idx, next_states, residuals, carry_extra
                )
                _, values, _ = self.agent.functional_act(
                    actor_state,
                    critic_state,
                    policy_input,
                    value_key,
                )
                return values, rng_key, carry_extra

            extra = AdaptationRolloutExtra(
                history=jnp.zeros((num_envs, history_len, state_action_dim)),
                counts=jnp.zeros((num_envs,), dtype=jnp.int32),
            )

            rollout_result = run_functional_rollout(
                step_config=step_config,
                initial_state=vec_state,
                initial_residuals=jnp.zeros((num_envs, self.env.res_dim)),
                rng=self.agent.key,
                num_steps=self.steps_per_epoch,
                reference_waypoint=self.reference_point,
                callbacks=FunctionalRolloutCallbacks(
                    prepare_policy_input=_prepare_policy_input,
                    sample_policy=_sample_policy,
                    post_step=_post_step,
                    bootstrap_value=_bootstrap_value,
                ),
                extra=extra,
            )

            jax.block_until_ready(rollout_result.actions)
            self.agent.key = rollout_result.final_rng

            step_outputs = rollout_result.step_outputs
            actions_traj = rollout_result.actions
            actual_wrench = step_outputs.actual_wrench
            next_obs = step_outputs.next_obs
            history_mask = rollout_result.aux.astype(bool)
            done_flags = rollout_result.done_flags

            state_action_data = jnp.concatenate(
                [step_outputs.prev_states, actions_traj], axis=2
            )
            extrinsics = actual_wrench

            state_action_data, extrinsics = self._build_sliding_windows(
                state_action_data,
                extrinsics,
                history_mask,
                history_len,
            )
            if state_action_data.shape[0] == 0:
                print(
                    "Skipping adaptation module update: insufficient full-history samples."
                )
                continue

            history = jnp.zeros((num_envs, history_len, state_action_dim))
            counts = jnp.zeros((num_envs,), dtype=jnp.int32)
            tracking_vals = []
            angle_vals = []
            extrinsic_vals = []

            for step_idx in range(self.steps_per_epoch):
                combined = jnp.concatenate(
                    [step_outputs.prev_states[step_idx], actions_traj[step_idx]], axis=1
                )
                history = jnp.roll(history, shift=-1, axis=1)
                history = history.at[:, -1, :].set(combined)
                counts = jnp.minimum(counts + 1, history_len)
                history_full = history_mask[step_idx]

                obs_step = next_obs[step_idx]
                actual_step = actual_wrench[step_idx]
                extrinsic_est = None
                if bool(jnp.all(history_full)):
                    extrinsic_est = self.adaptation_module(history)
                tracking_mean, angle_mean, extrinsic_mean = self._compute_mean_errors(
                    obs_step,
                    actual_step,
                    extrinsic_est,
                )
                tracking_vals.append(tracking_mean)
                angle_vals.append(angle_mean)
                extrinsic_vals.append(
                    0.0 if extrinsic_mean is None else float(extrinsic_mean)
                )

                if bool(done_flags[step_idx]) and step_idx != self.steps_per_epoch - 1:
                    history = jnp.zeros((num_envs, history_len, state_action_dim))
                    counts = jnp.zeros((num_envs,), dtype=jnp.int32)

            tracking_vals = jnp.asarray(tracking_vals)
            angle_vals = jnp.asarray(angle_vals)
            extrinsic_vals = jnp.asarray(extrinsic_vals)

            num_nn_epochs = 100
            split_key = self._take_keys()
            (
                X_train,
                y_train,
                X_val,
                y_val,
                _,
            ) = train_val_split(
                state_action_data,
                extrinsics,
                key=split_key,
                shuffle=False,
            )

            train_loss_sum = 0.0
            train_loss_count = 0
            val_loss_sum = 0.0
            val_loss_count = 0
            last_val_loss = None
            am_train_loss_value = 0.0

            for nn_epoch in range(num_nn_epochs):
                am_train_loss, grads = self.jitted_batched_am_loss_and_grad(
                    self.am,
                    X_train,
                    y_train,
                )
                am_train_loss_value = float(am_train_loss)
                train_loss_sum += am_train_loss_value
                train_loss_count += 1
                self.am_optimizer.update(grads)

                if nn_epoch % 10 == 0:
                    am_val_loss, _ = self.jitted_batched_am_loss_and_grad(
                        self.am, X_val, y_val
                    )
                    am_val_loss_value = float(am_val_loss)
                    val_loss_sum += am_val_loss_value
                    val_loss_count += 1
                    last_val_loss = am_val_loss_value

            mean_am_train_loss = (
                train_loss_sum / train_loss_count if train_loss_count > 0 else 0.0
            )
            mean_am_val_loss = (
                val_loss_sum / val_loss_count if val_loss_count > 0 else 0.0
            )

            if self.env.use_wandb:
                wandb.log(
                    {
                        "am_train_loss_last": am_train_loss_value,
                        "am_train_loss_mean": mean_am_train_loss,
                        "am_val_loss_last": (
                            last_val_loss if last_val_loss is not None else float("nan")
                        ),
                        "am_val_loss_mean": (
                            mean_am_val_loss if val_loss_count > 0 else float("nan")
                        ),
                    }
                )

            if self.agent.has_logger:
                self.env.logger.log(
                    self.env.run_id,
                    float(self.env.mjx_batch.time[0]),
                    step=int(epoch),
                    run_name=self.env.run_name,
                    stage="am_training",
                    am_train_loss_last=am_train_loss_value,
                    am_train_loss_mean=mean_am_train_loss,
                    am_val_loss_last=(
                        last_val_loss if last_val_loss is not None else 0.0
                    ),
                    am_val_loss_mean=mean_am_val_loss,
                    mean_tracking_error=float(tracking_vals.mean()),
                    mean_angle_error=float(angle_vals.mean()),
                    mean_extrinsic_error=float(extrinsic_vals.mean()),
                )

            save_adaptation_module(
                self.am, self.ckpt_dir, self.adaptation_module_file_name
            )

    def posttrain(self) -> None:
        """
        Fine-tune the policy on imperfectly estimated residual wrenches (phase 3 in A-RMA).
        NOTE: use_adaptive_approach must be set to True in the environment config.
        """
        raise NotImplementedError("Post-training is not implemented yet.\n")

    def evaluate(self, phase: int = 2) -> None:
        """
        Evaluate the agent.
        If phase == 1, evaluate base policy before training the adaptation module.
        If phase == 2, evaluate base policy after training the adaptation module.
        """
        # Check if trained actor, critic and adaptation modules are available and load them
        file_path = os.path.join(self.ckpt_dir, self.training_state_file_name)
        if os.path.isfile(file_path):
            restored_state = load_trained_modules(
                self.ckpt_dir, self.training_state_file_name
            )
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
            raise Exception("Not all necessary modules have been trained yet.\n")

        if phase not in (1, 2):
            raise ValueError("Phase must be either 1 or 2.")

        subkeys_eval = self._take_keys(self.n_evals)
        subkeys_eval = jnp.atleast_2d(subkeys_eval)

        num_envs = self.env.num_envs
        history_len = self.env.history_len
        state_action_dim = self.state_action_dim
        returns = jnp.zeros((num_envs, self.n_evals), dtype=jnp.float32)

        for eval_idx in range(self.n_evals):
            print(f"Testing policy: episode {eval_idx + 1}/{self.n_evals}")
            self.env.reset()
            self.env.reset_perturbations()

            if self.env.train_with_failures and eval_idx >= self.n_evals // 2:
                eval_key = subkeys_eval[eval_idx]
                self.env.apply_random_perturbations(
                    key=eval_key,
                    fraction_perturbed_envs=0.5,
                )
                self.env.apply_random_disturbance(
                    key=eval_key,
                    fraction_disturbed_envs=0.15,
                )

            step_config = self.env.build_step_config(
                max_episode_len=self.episode_len,
            )
            vec_state = self.env.state_struct

            if self.env.use_adaptive_approach:
                residual_init = jnp.zeros(
                    (num_envs, self.env.res_dim), dtype=jnp.float32
                )
            else:
                residual_init = jnp.zeros((num_envs, 0), dtype=jnp.float32)

            extra = AdaptationRolloutExtra(
                history=jnp.zeros((num_envs, history_len, state_action_dim)),
                counts=jnp.zeros((num_envs,), dtype=jnp.int32),
            )

            def _prepare_policy_input(_step, states, residuals, carry_extra):
                del _step  # unused
                if residuals.shape[-1]:
                    return jnp.concatenate([states, residuals], axis=1), carry_extra
                return states, carry_extra

            def _sample_policy(_step, policy_input, rng_key, carry_extra):
                del _step  # unused
                actions = self.agent.get_control_input("evaluation", policy_input)
                zeros = jnp.zeros((num_envs,), dtype=jnp.float32)
                return actions, zeros, zeros, rng_key, carry_extra

            def _post_step(
                _step, step_output, actions, residuals, reset_flag, carry_extra
            ):
                del _step, residuals  # unused
                history = carry_extra.history
                counts = carry_extra.counts

                combined = jnp.concatenate([step_output.prev_states, actions], axis=1)
                history = jnp.roll(history, shift=-1, axis=1)
                history = history.at[:, -1, :].set(combined)
                counts = jnp.minimum(counts + 1, history_len)

                def _reset_hist(_):
                    return (
                        jnp.zeros_like(history),
                        jnp.zeros_like(counts),
                    )

                history, counts = jax.lax.cond(
                    reset_flag,
                    _reset_hist,
                    lambda _: (history, counts),
                    operand=None,
                )

                if self.env.use_adaptive_approach:
                    desired = step_output.desired_wrench
                    if phase == 1:
                        extrinsic = step_output.actual_wrench
                    else:  # phase == 2 guaranteed by earlier check
                        extrinsic = self.adaptation_module(history)
                    residuals_next = extrinsic - desired
                else:
                    residuals_next = jnp.zeros(
                        (num_envs, 0), dtype=step_output.actual_wrench.dtype
                    )

                history_full = counts >= history_len
                new_extra = AdaptationRolloutExtra(history=history, counts=counts)
                return residuals_next, history_full, new_extra

            def _bootstrap_value(_step, env_state, residuals, rng_key, carry_extra):
                del _step, env_state, residuals  # unused
                zeros = jnp.zeros((num_envs,), dtype=jnp.float32)
                return zeros, rng_key, carry_extra

            rollout_result = run_functional_rollout(
                step_config=step_config,
                initial_state=vec_state,
                initial_residuals=residual_init,
                rng=self.agent.key,
                num_steps=self.episode_len,
                reference_waypoint=self.reference_point,
                callbacks=FunctionalRolloutCallbacks(
                    prepare_policy_input=_prepare_policy_input,
                    sample_policy=_sample_policy,
                    post_step=_post_step,
                    bootstrap_value=_bootstrap_value,
                ),
                extra=extra,
            )

            jax.block_until_ready(rollout_result.actions)
            self.agent.key = rollout_result.final_rng

            self.env._state = rollout_result.final_state
            self.env._rng = rollout_result.final_state.rng
            self.env.mjx_batch = rollout_result.final_state.mjx_batch
            self.env.disturbance_states = rollout_result.final_state.disturbance_states
            self.env.perturbation_states = (
                rollout_result.final_state.perturbation_states
            )
            if hasattr(self.env, "disturbances") and self.env.disturbances is not None:
                for obj, snapshot in zip(
                    self.env.disturbances.disturbances,
                    rollout_result.final_state.disturbance_states,
                    strict=True,
                ):
                    obj.state = snapshot
            if (
                hasattr(self.env, "perturbations")
                and self.env.perturbations is not None
            ):
                for obj, snapshot in zip(
                    self.env.perturbations.perturbations,
                    rollout_result.final_state.perturbation_states,
                    strict=True,
                ):
                    obj.state = snapshot
            self.env._refresh_effect_states()

            step_outputs = rollout_result.step_outputs
            rewards = step_outputs.rewards
            done_flags = rollout_result.done_flags
            residuals_traj = rollout_result.residuals

            done_indices = jnp.nonzero(
                done_flags, size=self.episode_len, fill_value=-1
            )[0]
            done_indices = jax.device_get(done_indices)
            done_idx = (
                int(done_indices[0])
                if done_indices.size and done_indices[0] >= 0
                else self.episode_len - 1
            )
            valid_slice = slice(0, done_idx + 1)

            rewards_slice = rewards[valid_slice]
            returns_eval = jnp.sum(rewards_slice, axis=0)
            returns = returns.at[:, eval_idx].set(returns_eval)

            tracking_vals = []
            angle_vals = []
            extrinsic_vals = []

            for step_id in range(done_idx + 1):
                obs_step = step_outputs.next_obs[step_id]
                actual_step = step_outputs.actual_wrench[step_id]
                if self.env.use_adaptive_approach:
                    if phase == 1:
                        extr_step = actual_step
                    else:
                        extr_step = (
                            residuals_traj[step_id]
                            + step_outputs.desired_wrench[step_id]
                        )
                else:
                    extr_step = None

                tracking_step, angle_step, extr_step_error = self._compute_mean_errors(
                    obs_step, actual_step, extr_step
                )
                tracking_vals.append(tracking_step)
                angle_vals.append(angle_step)
                if extr_step_error is not None:
                    extrinsic_vals.append(extr_step_error)

            tracking_mean = (
                float(jnp.stack(tracking_vals).mean()) if tracking_vals else 0.0
            )
            angle_mean = float(jnp.stack(angle_vals).mean()) if angle_vals else 0.0
            mean_extrinsic_error = (
                float(jnp.stack(extrinsic_vals).mean()) if extrinsic_vals else 0.0
            )
            terminal_count = float(step_outputs.terminals[valid_slice].sum())

            if self.agent.has_logger:
                self.env.logger.log(
                    self.env.run_id,
                    float(self.env.mjx_batch.time[0]),
                    step=int(eval_idx),
                    run_name=self.env.run_name,
                    stage="evaluation",
                    mean_episodic_returns=float(returns_eval.mean()),
                    num_terminal=terminal_count,
                    mean_tracking_error=tracking_mean,
                    mean_angle_error=angle_mean,
                    mean_extrinsic_error=mean_extrinsic_error,
                )

        print(
            f"Average episodic return over all evals and all envs: {float(returns.mean())}\n"
        )

    def _generate_experience(self) -> None:
        """
        Roll out an episode where the actions are computed from a PD controller that serves as training data.
        """
        # Check if pretraining data already exists
        file_path = os.path.join(self.ckpt_dir, self.pretraining_data_file_name)
        if os.path.isfile(file_path):
            return

        print("Gather training data using the PD controller...\n")

        # Set up buffer
        buffer = ReplayBuffer(
            self.env.num_envs,
            self.env.obs_dim,
            self.env.act_dim,
            self.env.res_dim,
            self.steps_per_epoch,
            self.gamma,
            self.lam,
        )

        # Initialize the environment
        self.env.reset()
        states, ep_ret, ep_len = (
            self.env.get_states(self.reference_point),
            jnp.zeros(self.env.num_envs),
            0,
        )

        if self.env.use_adaptive_approach is True:
            res = jnp.zeros((self.env.num_envs, self.env.res_dim))
        else:
            res = jnp.empty((self.env.num_envs, 0))

        # Epoch variables
        ep_returns = jnp.zeros((self.env.num_envs, self.steps_per_epoch))

        # Main training loop
        for t in range(self.steps_per_epoch):
            # Get value estimates from the agent
            _, v, logp = self.agent.act(
                jnp.concatenate([states, res], axis=1)
            )  # Use un-normalized states
            self.env.obs = self.env.get_obs()

            # Compute actions from the PD controller
            a = self.pd_ctrl.get_control_input(self.env)

            # Perform environment transition
            r, terminal = self.env.transition(
                a, jnp.concatenate([states, res], axis=1), self.reference_point
            )
            ep_returns = ep_returns.at[:, t].set(self.gamma * ep_returns[:, t - 1] + r)
            ep_ret += r
            ep_len += 1

            # Log transition
            buffer.store(states, a, r, v, logp, res)  # Use un-normalized states

            # Update state
            states = self.env.get_states(self.reference_point)

            # Update extrinsics
            if self.env.use_adaptive_approach is True:
                actual_wrench = self.env.get_actual_wrench()
                desired_wrench = self.env.get_desired_wrench(a)
                res = actual_wrench - desired_wrench
            else:
                res = jnp.empty((self.env.num_envs, 0))

            # Check if a timeout is appropriate
            timeout = ep_len == self.max_ep_len
            epoch_ended = t == self.steps_per_epoch - 1

            # N. B.: could also have a different ep_len for each env and consider timeout and terminal conditions
            # for each env individually
            if terminal.all() or timeout or epoch_ended:
                # If the trajectory didn't reach terminal state, bootstrap value target
                if epoch_ended:
                    _, v, _ = self.agent.act(
                        jnp.concatenate([states, res], axis=1)
                    )  # Use un-normalized states
                else:
                    v = jnp.zeros(self.env.num_envs)

                buffer.end_traj(v)

                self.env.reset()
                states, ep_ret, ep_len = (
                    self.env.get_states(self.reference_point),
                    jnp.zeros(self.env.num_envs),
                    0,
                )

                if self.env.use_adaptive_approach is True:
                    res = jnp.zeros((self.env.num_envs, self.env.res_dim))
                else:
                    res = jnp.empty((self.env.num_envs, 0))

        # Get the data from the training loop and save it
        data = buffer.get()
        save_training_data(self.ckpt_dir, self.pretraining_data_file_name, data)

    def _build_adaptation_module(self):
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

    def _select_am_loss_fn(self):
        if self.env.am_architecture == "transformer":
            kl_weight = float(self.am_kl_weight)

            def _loss(model, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
                mu, log_sigma = jax.vmap(lambda hist: model(hist, return_stats=True))(X)
                target = y[:, -1, :]
                log_sigma = jnp.clip(log_sigma, -6.0, 2.0)
                sigma_sq = jnp.exp(2.0 * log_sigma)
                nll = 0.5 * jnp.sum(
                    ((target - mu) ** 2) / sigma_sq + 2.0 * log_sigma, axis=-1
                )
                if kl_weight > 0.0:
                    kl = 0.5 * jnp.sum(
                        mu**2 + sigma_sq - 1.0 - jnp.log(sigma_sq + 1e-8), axis=-1
                    )
                    return jnp.mean(nll + kl_weight * kl)
                return jnp.mean(nll)

            return _loss

        def _mse_loss(model, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            preds = jax.vmap(model)(X)
            target = y[:, -1, :]
            mse = jnp.square(preds - target)
            return jnp.mean(jnp.sum(mse, axis=-1))

        return _mse_loss

    def _build_sliding_windows(
        self,
        features: jnp.ndarray,
        targets: jnp.ndarray,
        mask: jnp.ndarray,
        seq_len: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Construct stride-1 windows of length ``seq_len`` and keep only entries with a fully populated history.
        """
        # Discard early timesteps where the history buffer is still warming up.
        valid_mask = mask.at[: seq_len - 1, :].set(False)
        valid_count = int(valid_mask.sum())
        if valid_count == 0:
            return (
                jnp.empty((0, seq_len, features.shape[2]), dtype=features.dtype),
                jnp.empty((0, seq_len, targets.shape[2]), dtype=targets.dtype),
            )

        # Flatten (time, env) indices where a full history is available.
        valid_t, valid_env = jnp.where(
            valid_mask,
            size=int(valid_mask.size),
            fill_value=-1,
        )
        valid_t = valid_t[:valid_count]
        valid_env = valid_env[:valid_count]
        starts = valid_t - (seq_len - 1)

        feat_dim = features.shape[2]
        target_dim = targets.shape[2]

        def _slice_single(start: jnp.ndarray, env_idx: jnp.ndarray):
            start = jnp.asarray(start, dtype=jnp.int32)
            env_idx = jnp.asarray(env_idx, dtype=jnp.int32)
            feat_slice = jax.lax.dynamic_slice(
                features,
                (start, env_idx, 0),
                (seq_len, 1, feat_dim),
            )
            feat_slice = jnp.squeeze(feat_slice, axis=1)
            target_slice = jax.lax.dynamic_slice(
                targets,
                (start, env_idx, 0),
                (seq_len, 1, target_dim),
            )
            target_slice = jnp.squeeze(target_slice, axis=1)
            return feat_slice, target_slice

        feat_windows, target_windows = jax.vmap(_slice_single)(starts, valid_env)
        return feat_windows, target_windows

    def _compute_mean_errors(
        self,
        obs: jnp.ndarray,
        actual_wrench: jnp.ndarray | None = None,
        ext: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
        """
        Compute mean tracking, attitude, and extrinsic errors for the current step.
        """

        tracking = calc_lateral_tracking_error(obs, self.planner)
        attitude = jnp.degrees(calc_attitude_error(obs))

        tracking_mean = tracking.mean()
        attitude_mean = attitude.mean()

        extrinsic_mean = None
        if (
            self.env.use_adaptive_approach
            and actual_wrench is not None
            and ext is not None
        ):
            extrinsic_error = calc_extrinsic_error(ext, actual_wrench)
            extrinsic_mean = extrinsic_error.mean()

        return tracking_mean, attitude_mean, extrinsic_mean

    def _load_rl_hyperparams(self) -> None:
        """
        Load the relevant hyperparams from the config file.
        """
        if isinstance(self.agent, VPG):
            self.agent._load_vpg_hyperparams()
        elif isinstance(self.agent, PPO):
            self.agent._load_ppo_hyperparams()
        else:
            raise Exception("Agent has not been implemented.")

        # Mirror agent hyperparameters locally for convenience
        self.steps_per_epoch = self.agent.steps_per_epoch
        self.epochs = self.agent.epochs
        self.max_ep_len = self.agent.max_ep_len
        self.gamma = self.agent.gamma
        self.lam = self.agent.lam

        self.episode_len = self.env.env_cfg.control.RL.episode_len
        self.n_evals = self.env.env_cfg.control.RL.n_evals

    def _create_checkpoint_file_names(self) -> None:
        """
        Create the checkpoint file names for the pretraining and training data and states.
        """

        # Build filename components
        adaptive = "adaptive" if self.env.use_adaptive_approach else None
        pretrained = "pretrained" if self.env.use_pretrained else None
        nominal = None if self.env.train_with_failures else "nominal"

        def build_name(prefix: str) -> str:
            parts = [prefix, adaptive, pretrained, nominal]
            return "_".join(p for p in parts if p) + ".pkl"

        # Pretraining filenames
        if adaptive:
            self.pretraining_data_file_name = "pretraining_data_adaptive.pkl"
            self.pretraining_state_file_name = "pretraining_state_adaptive.pkl"
        else:
            self.pretraining_data_file_name = "pretraining_data.pkl"
            self.pretraining_state_file_name = "pretraining_state.pkl"

        # Training filenames
        self.training_data_file_name = build_name("training_data")
        self.training_state_file_name = build_name("training_state")
        self.adaptation_module_file_name = (
            f"adapt_module_state_{self.env.am_architecture}.pkl"
        )
