from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from smallsat_sim.controllers.rl.runners import rollout as ru
from smallsat_sim.controllers.rl.runners.adaptive_context import (
    build_adaptation_query,
    build_adaptive_context,
    estimate_thruster_effectiveness,
)
from smallsat_sim.controllers.rl.storage.replay_buffer import ReplayBuffer


@dataclass
class DummyBatch:
    qpos: jnp.ndarray

    def replace(self, **updates: jnp.ndarray) -> "DummyBatch":
        return DummyBatch(qpos=updates.get("qpos", self.qpos))


def _dummy_batch_flatten(batch: DummyBatch):
    return (batch.qpos,), None


def _dummy_batch_unflatten(aux_data, children):
    (qpos,) = children
    return DummyBatch(qpos=qpos)


jax.tree_util.register_pytree_node(DummyBatch, _dummy_batch_flatten, _dummy_batch_unflatten)


@dataclass
class DummyState:
    rng: jnp.ndarray
    mjx_batch: DummyBatch

    def replace(self, **updates):
        return DummyState(
            rng=updates.get("rng", self.rng),
            mjx_batch=updates.get("mjx_batch", self.mjx_batch),
        )


def _dummy_state_flatten(state: DummyState):
    return (state.rng, state.mjx_batch), None


def _dummy_state_unflatten(aux_data, children):
    rng, mjx_batch = children
    return DummyState(rng=rng, mjx_batch=mjx_batch)


jax.tree_util.register_pytree_node(DummyState, _dummy_state_flatten, _dummy_state_unflatten)


@dataclass
class DummyStepConfig:
    max_episode_len: int


@dataclass
class DummyStepOutput:
    prev_states: jnp.ndarray
    next_states: jnp.ndarray
    rewards: jnp.ndarray
    terminals: jnp.ndarray
    commanded_ctrl: jnp.ndarray
    applied_ctrl: jnp.ndarray
    actual_wrench: jnp.ndarray
    desired_wrench: jnp.ndarray
    prev_obs: jnp.ndarray
    next_obs: jnp.ndarray
    success_terminals: jnp.ndarray
    failure_terminals: jnp.ndarray
    reward_components: dict


def _dummy_step_output_flatten(output: DummyStepOutput):
    children = (
        output.prev_states,
        output.next_states,
        output.rewards,
        output.terminals,
        output.commanded_ctrl,
        output.applied_ctrl,
        output.actual_wrench,
        output.desired_wrench,
        output.prev_obs,
        output.next_obs,
        output.success_terminals,
        output.failure_terminals,
        output.reward_components,
    )
    return children, None


def _dummy_step_output_unflatten(aux_data, children):
    (
        prev_states,
        next_states,
        rewards,
        terminals,
        commanded_ctrl,
        applied_ctrl,
        actual_wrench,
        desired_wrench,
        prev_obs,
        next_obs,
        success_terminals,
        failure_terminals,
        reward_components,
    ) = children
    return DummyStepOutput(
        prev_states=prev_states,
        next_states=next_states,
        rewards=rewards,
        terminals=terminals,
        commanded_ctrl=commanded_ctrl,
        applied_ctrl=applied_ctrl,
        actual_wrench=actual_wrench,
        desired_wrench=desired_wrench,
        prev_obs=prev_obs,
        next_obs=next_obs,
        success_terminals=success_terminals,
        failure_terminals=failure_terminals,
        reward_components=reward_components,
    )


jax.tree_util.register_pytree_node(
    DummyStepOutput,
    _dummy_step_output_flatten,
    _dummy_step_output_unflatten,
)


def _dummy_step_config(
    *,
    num_envs: int,
    max_episode_len: int,
    res_dim: int = 0,
    use_adaptive_approach: bool = False,
):
    del num_envs, res_dim, use_adaptive_approach
    return DummyStepConfig(max_episode_len=max_episode_len)


def test_compute_terminals_radius_threshold() -> None:
    vec_env = pytest.importorskip("smallsat_sim.envs.vec_env")
    states = jnp.zeros((2, 12), dtype=jnp.float32)
    states = states.at[0, 0].set(0.1)
    states = states.at[1, 0].set(0.31)
    hold_counts = jnp.zeros((2,), dtype=jnp.int32)

    config = SimpleNamespace(
        terminal_radius=0.3,
        terminal_max_speed=10.0,
        terminal_max_att_error=10.0,
        terminal_max_ang_speed=10.0,
        terminal_hold_steps=1,
    )
    terminals, counts = vec_env._compute_terminals(states, hold_counts, config)

    assert bool(terminals[0])
    assert not bool(terminals[1])
    assert int(counts[0]) == 1
    assert int(counts[1]) == 0


def test_compute_terminals_requires_consecutive_hold_steps() -> None:
    vec_env = pytest.importorskip("smallsat_sim.envs.vec_env")
    states = jnp.zeros((1, 12), dtype=jnp.float32)
    config = SimpleNamespace(
        terminal_radius=0.3,
        terminal_max_speed=10.0,
        terminal_max_att_error=10.0,
        terminal_max_ang_speed=10.0,
        terminal_hold_steps=3,
    )

    terminals_1, counts_1 = vec_env._compute_terminals(
        states, jnp.array([0], dtype=jnp.int32), config
    )
    terminals_2, counts_2 = vec_env._compute_terminals(states, counts_1, config)
    terminals_3, counts_3 = vec_env._compute_terminals(states, counts_2, config)

    assert not bool(terminals_1[0])
    assert not bool(terminals_2[0])
    assert bool(terminals_3[0])
    assert int(counts_3[0]) == 3


def test_compute_terminals_requires_full_pose_success() -> None:
    vec_env = pytest.importorskip("smallsat_sim.envs.vec_env")
    states = jnp.zeros((3, 12), dtype=jnp.float32)
    states = states.at[1, 3].set(100.0)
    states = states.at[2, 6].set(100.0)
    states = states.at[2, 9].set(100.0)
    config = SimpleNamespace(
        terminal_radius=0.3,
        terminal_max_speed=0.15,
        terminal_max_att_error=0.25,
        terminal_max_ang_speed=0.05,
        terminal_hold_steps=1,
    )

    terminals, counts = vec_env._compute_terminals(
        states, jnp.array([0, 0, 0], dtype=jnp.int32), config
    )

    assert bool(terminals[0])
    assert not bool(terminals[1])
    assert bool(terminals[2])
    assert int(counts[0]) == 1
    assert int(counts[1]) == 0
    assert int(counts[2]) == 1


def test_compute_penalties_residual_clip_and_terminal_terms() -> None:
    vec_env = pytest.importorskip("smallsat_sim.envs.vec_env")
    prev_states = jnp.zeros((2, 12), dtype=jnp.float32)
    next_states = prev_states.at[:, 6].set(1.0)
    terminals = jnp.array([True, False])
    commanded_ctrl = jnp.array([[1.0, -1.0], [0.5, 0.5]], dtype=jnp.float32)
    prev_residuals = jnp.array([[3.0, 4.0], [0.0, 0.0]], dtype=jnp.float32)

    config = vec_env.VecEnvStepConfig(
        mjx_model=None,
        mjx_data_template=None,
        mjx_batch_template=DummyBatch(qpos=jnp.zeros((2, 1), dtype=jnp.float32)),
        init_qpos=jnp.zeros((2, 1), dtype=jnp.float32),
        init_qvel=jnp.zeros((2, 1), dtype=jnp.float32),
        num_envs=2,
        max_start_offset=0.0,
        control_decimation=1,
        sigma_pos=1.0,
        sigma_vel=1.0,
        sigma_att=1.0,
        sigma_angvel=1.0,
        w_pos=1.0,
        w_vel=1.0,
        w_att=1.0,
        w_angvel=1.0,
        lam_fuel=1.0,
        lam_speed_terminal=2.0,
        lam_ang_speed_terminal=0.0,
        lam_fuel_terminal=3.0,
        terminal_bonus=1.0,
        terminal_radius=0.3,
        terminal_max_speed=10.0,
        terminal_max_att_error=10.0,
        terminal_max_ang_speed=10.0,
        terminal_hold_steps=1,
        enable_failure_termination=False,
        failure_max_position_error=100.0,
        failure_max_speed=100.0,
        failure_max_att_error=100.0,
        failure_max_ang_speed=100.0,
        lam_wrench_residual=4.0,
        wrench_residual_tolerance=1.0,
        wrench_residual_clip=2.0,
        max_episode_len=8,
        res_dim=2,
        use_adaptive_approach=True,
        collect_reward_components=False,
        thruster_mixer_T=jnp.zeros((1, 1), dtype=jnp.float32),
        base_disturbance_states=(),
        base_perturbation_states=(),
    )

    penalties, details = vec_env._compute_penalties(
        prev_states,
        next_states,
        terminals,
        commanded_ctrl,
        prev_residuals,
        config,
    )

    expected = jnp.array([18.0, 1.0], dtype=jnp.float32)
    assert jnp.allclose(penalties, expected)
    assert jnp.allclose(details["penalty_fuel"], jnp.array([2.0, 1.0]))
    assert jnp.allclose(details["penalty_terminal_speed"], jnp.array([2.0, 0.0]))
    assert jnp.allclose(details["penalty_terminal_fuel"], jnp.array([6.0, 0.0]))
    assert jnp.allclose(details["penalty_wrench_residual"], jnp.array([8.0, 0.0]))


def test_run_functional_rollout_resets_and_reports_returns() -> None:
    num_envs = 2
    num_steps = 3

    initial_batch = DummyBatch(qpos=jnp.zeros((num_envs, 1), dtype=jnp.float32))
    initial_state = DummyState(
        rng=jax.random.PRNGKey(0),
        mjx_batch=initial_batch,
    )
    step_config = _dummy_step_config(num_envs=num_envs, max_episode_len=10)
    initial_residuals = jnp.zeros((num_envs, 0), dtype=jnp.float32)
    reference_waypoint = jnp.zeros((3,), dtype=jnp.float32)

    def _vecenv_step_stub(state, actions, _waypoint, _config, _prev_residuals):
        qpos = state.mjx_batch.qpos
        terminals = qpos[:, 0] == 0.0
        next_batch = state.mjx_batch.replace(qpos=qpos + 1.0)
        next_state = state.replace(mjx_batch=next_batch)
        rewards = jnp.full((num_envs,), 2.0, dtype=jnp.float32)
        step_output = DummyStepOutput(
            prev_states=qpos,
            next_states=qpos + 1.0,
            rewards=rewards,
            terminals=terminals,
            commanded_ctrl=actions,
            applied_ctrl=actions,
            actual_wrench=jnp.zeros((num_envs, 1), dtype=jnp.float32),
            desired_wrench=jnp.zeros((num_envs, 1), dtype=jnp.float32),
            prev_obs=qpos,
            next_obs=qpos + 1.0,
            success_terminals=terminals,
            failure_terminals=jnp.zeros_like(terminals),
            reward_components={},
        )
        return next_state, step_output

    def _vecenv_reset_stub(state, _config):
        reset_qpos = jnp.full_like(state.mjx_batch.qpos, -1.0)
        return state.replace(mjx_batch=state.mjx_batch.replace(qpos=reset_qpos))

    def _prepare(_step, states, residuals, carry_extra):
        del residuals
        return states, carry_extra

    def _sample(_step, policy_input, rng_key, carry_extra):
        actions = jnp.zeros((num_envs, 1), dtype=jnp.float32)
        values = jnp.ones((num_envs,), dtype=jnp.float32)
        logp = jnp.zeros((num_envs,), dtype=jnp.float32)
        return actions, values, logp, rng_key, carry_extra

    def _post(_step, step_output, actions, residuals, reset_flag, carry_extra):
        del step_output, actions, reset_flag
        return residuals, None, carry_extra

    def _bootstrap(_step, env_state, residuals, rng_key, carry_extra):
        del env_state, residuals
        return jnp.zeros((num_envs,), dtype=jnp.float32), rng_key, carry_extra

    result = ru.run_functional_rollout(
        step_config=step_config,
        initial_state=initial_state,
        initial_residuals=initial_residuals,
        rng=initial_state.rng,
        num_steps=num_steps,
        reference_waypoint=reference_waypoint,
        callbacks=ru.FunctionalRolloutCallbacks(
            prepare_policy_input=_prepare,
            sample_policy=_sample,
            post_step=_post,
            bootstrap_value=_bootstrap,
        ),
        state_features_fn=lambda batch, _: batch.qpos,
        step_fn=_vecenv_step_stub,
        reset_fn=_vecenv_reset_stub,
    )

    assert bool(result.done_flags[0])
    assert bool(result.done_masks[0].all())
    assert bool(result.terminated_masks[0].all())
    assert not bool(result.truncated_masks[0].any())
    assert jnp.allclose(result.episode_returns[0], jnp.full((num_envs,), 2.0))
    assert jnp.allclose(result.episode_returns[1], jnp.zeros((num_envs,)))
    assert jnp.allclose(result.episode_returns[2], jnp.full((num_envs,), 4.0))
    assert jnp.allclose(result.final_state.mjx_batch.qpos, jnp.ones((num_envs, 1)))


def test_run_functional_rollout_bootstraps_timeouts() -> None:
    num_envs = 2
    num_steps = 5
    max_episode_len = 2

    initial_batch = DummyBatch(qpos=jnp.zeros((num_envs, 1), dtype=jnp.float32))
    initial_state = DummyState(
        rng=jax.random.PRNGKey(0),
        mjx_batch=initial_batch,
    )
    step_config = _dummy_step_config(num_envs=num_envs, max_episode_len=max_episode_len)
    initial_residuals = jnp.zeros((num_envs, 0), dtype=jnp.float32)
    reference_waypoint = jnp.zeros((3,), dtype=jnp.float32)

    def _vecenv_step_stub(state, actions, _waypoint, _config, _prev_residuals):
        qpos = state.mjx_batch.qpos
        terminals = jnp.zeros((num_envs,), dtype=bool)
        next_batch = state.mjx_batch.replace(qpos=qpos + 1.0)
        next_state = state.replace(mjx_batch=next_batch)
        rewards = jnp.ones((num_envs,), dtype=jnp.float32)
        step_output = DummyStepOutput(
            prev_states=qpos,
            next_states=qpos + 1.0,
            rewards=rewards,
            terminals=terminals,
            commanded_ctrl=actions,
            applied_ctrl=actions,
            actual_wrench=jnp.zeros((num_envs, 1), dtype=jnp.float32),
            desired_wrench=jnp.zeros((num_envs, 1), dtype=jnp.float32),
            prev_obs=qpos,
            next_obs=qpos + 1.0,
            success_terminals=terminals,
            failure_terminals=jnp.zeros_like(terminals),
            reward_components={},
        )
        return next_state, step_output

    def _vecenv_reset_stub(state, _config):
        reset_qpos = jnp.zeros_like(state.mjx_batch.qpos)
        return state.replace(mjx_batch=state.mjx_batch.replace(qpos=reset_qpos))

    def _prepare(_step, states, residuals, carry_extra):
        del residuals
        return states, carry_extra

    def _sample(_step, policy_input, rng_key, carry_extra):
        actions = jnp.zeros((num_envs, 1), dtype=jnp.float32)
        values = jnp.ones((num_envs,), dtype=jnp.float32)
        logp = jnp.zeros((num_envs,), dtype=jnp.float32)
        return actions, values, logp, rng_key, carry_extra

    def _post(_step, step_output, actions, residuals, reset_flag, carry_extra):
        del step_output, actions, reset_flag
        return residuals, None, carry_extra

    def _bootstrap(_step, env_state, residuals, rng_key, carry_extra):
        del env_state, residuals, carry_extra
        return jnp.full((num_envs,), 7.0, dtype=jnp.float32), rng_key, None

    result = ru.run_functional_rollout(
        step_config=step_config,
        initial_state=initial_state,
        initial_residuals=initial_residuals,
        rng=initial_state.rng,
        num_steps=num_steps,
        reference_waypoint=reference_waypoint,
        callbacks=ru.FunctionalRolloutCallbacks(
            prepare_policy_input=_prepare,
            sample_policy=_sample,
            post_step=_post,
            bootstrap_value=_bootstrap,
        ),
        state_features_fn=lambda batch, _: batch.qpos,
        step_fn=_vecenv_step_stub,
        reset_fn=_vecenv_reset_stub,
    )

    bootstrap_vals = result.bootstrap_values
    assert jnp.all(result.terminated_masks == 0)
    assert jnp.all(result.truncated_masks[1])
    assert jnp.all(result.truncated_masks[3])
    assert not bool(result.truncated_masks[0].any())
    assert not bool(result.truncated_masks[2].any())
    assert not bool(result.truncated_masks[4].any())
    assert jnp.allclose(bootstrap_vals[1], 7.0)
    assert jnp.allclose(bootstrap_vals[3], 7.0)
    assert jnp.allclose(bootstrap_vals[4], 7.0)
    assert jnp.allclose(bootstrap_vals[0], 0.0)
    assert jnp.allclose(bootstrap_vals[2], 0.0)


def test_run_functional_rollout_records_policy_input_residuals() -> None:
    num_envs = 1
    num_steps = 3

    initial_batch = DummyBatch(qpos=jnp.zeros((num_envs, 1), dtype=jnp.float32))
    initial_state = DummyState(
        rng=jax.random.PRNGKey(0),
        mjx_batch=initial_batch,
    )
    step_config = _dummy_step_config(num_envs=num_envs, max_episode_len=10)
    initial_residuals = jnp.zeros((num_envs, 1), dtype=jnp.float32)
    reference_waypoint = jnp.zeros((3,), dtype=jnp.float32)

    def _vecenv_step_stub(state, actions, _waypoint, _config, _prev_residuals):
        qpos = state.mjx_batch.qpos
        terminals = jnp.zeros((num_envs,), dtype=bool)
        next_state = state.replace(mjx_batch=state.mjx_batch.replace(qpos=qpos + 1.0))
        step_output = DummyStepOutput(
            prev_states=qpos,
            next_states=qpos + 1.0,
            rewards=jnp.ones((num_envs,), dtype=jnp.float32),
            terminals=terminals,
            commanded_ctrl=actions,
            applied_ctrl=actions,
            actual_wrench=jnp.zeros((num_envs, 1), dtype=jnp.float32),
            desired_wrench=jnp.zeros((num_envs, 1), dtype=jnp.float32),
            prev_obs=qpos,
            next_obs=qpos + 1.0,
            success_terminals=terminals,
            failure_terminals=jnp.zeros_like(terminals),
            reward_components={},
        )
        return next_state, step_output

    def _sample(_step, policy_input, rng_key, carry_extra):
        actions = policy_input
        values = jnp.zeros((num_envs,), dtype=jnp.float32)
        logp = jnp.zeros((num_envs,), dtype=jnp.float32)
        return actions, values, logp, rng_key, carry_extra

    def _post(_step, step_output, actions, residuals, reset_flag, carry_extra):
        del step_output, actions, reset_flag
        return residuals + 1.0, None, carry_extra

    result = ru.run_functional_rollout(
        step_config=step_config,
        initial_state=initial_state,
        initial_residuals=initial_residuals,
        rng=initial_state.rng,
        num_steps=num_steps,
        reference_waypoint=reference_waypoint,
        callbacks=ru.FunctionalRolloutCallbacks(
            prepare_policy_input=lambda _step, _states, residuals, extra: (
                residuals,
                extra,
            ),
            sample_policy=_sample,
            post_step=_post,
            bootstrap_value=ru.make_zero_bootstrap_value(num_envs),
        ),
        state_features_fn=lambda batch, _: batch.qpos,
        step_fn=_vecenv_step_stub,
        reset_fn=lambda state, _config: state,
    )

    expected = jnp.array([[[0.0]], [[1.0]], [[2.0]]], dtype=jnp.float32)
    assert jnp.allclose(result.actions, expected)
    assert jnp.allclose(result.residuals, expected)
    assert jnp.allclose(result.final_residuals, jnp.array([[3.0]], dtype=jnp.float32))


def test_update_history_buffer_reports_full_before_reset() -> None:
    extra = ru.AdaptationRolloutExtra(
        history=jnp.zeros((1, 2, 2), dtype=jnp.float32),
        counts=jnp.array([1], dtype=jnp.int32),
    )

    history, counts, history_full, new_extra = ru.update_history_buffer(
        carry_extra=extra,
        prev_states=jnp.array([[2.0]], dtype=jnp.float32),
        actions=jnp.array([[3.0]], dtype=jnp.float32),
        reset_flag=jnp.array([True]),
        history_len=2,
    )

    assert bool(history_full[0])
    assert jnp.allclose(history, jnp.zeros_like(history))
    assert jnp.allclose(counts, jnp.zeros_like(counts))
    assert jnp.allclose(new_extra.history, jnp.zeros_like(new_extra.history))


def test_am_history_contains_only_state_and_requested_action() -> None:
    extra = ru.AdaptationRolloutExtra(
        history=jnp.zeros((1, 3, 4), dtype=jnp.float32),
        counts=jnp.array([0], dtype=jnp.int32),
    )
    prev_states = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
    requested_actions = jnp.array([[3.0, 4.0]], dtype=jnp.float32)

    history, _, _, _ = ru.update_history_buffer(
        carry_extra=extra,
        prev_states=prev_states,
        actions=requested_actions,
        reset_flag=jnp.array([False]),
        history_len=3,
    )

    assert jnp.allclose(
        history[:, -1, :],
        jnp.concatenate([prev_states, requested_actions], axis=1),
    )


def test_am_query_contains_only_state_and_requested_wrench() -> None:
    states = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
    requested_wrench = jnp.array([[3.0, 4.0, 5.0]], dtype=jnp.float32)

    query = build_adaptation_query(
        states=states,
        desired_wrench=requested_wrench,
        use_task_conditioned_am=True,
    )

    assert jnp.allclose(query, jnp.concatenate([states, requested_wrench], axis=1))


def test_structured_adaptive_context_includes_bias_and_authority() -> None:
    commanded = jnp.array([[1.0, 1.0]], dtype=jnp.float32)
    applied = jnp.array([[0.5, 0.0]], dtype=jnp.float32)
    mixer_t = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    actual = applied @ mixer_t
    desired = commanded @ mixer_t
    previous = jnp.zeros((1, 7), dtype=jnp.float32)

    context = build_adaptive_context(
        commanded_ctrl=commanded,
        applied_ctrl=applied,
        actual_wrench=actual,
        desired_wrench=desired,
        previous_context=previous,
        use_adaptive_approach=True,
        adaptive_context_mode="structured",
        thruster_mixer_T=mixer_t,
    )

    assert context.shape == previous.shape
    assert jnp.allclose(context[:, :3], actual - desired)
    assert jnp.allclose(context[:, 3:6], 0.1 * (actual - desired))
    expected_normalized_error = jnp.linalg.norm(actual - desired, axis=1) / (
        jnp.linalg.norm(desired, axis=1) + 1e-6
    )
    assert jnp.allclose(context[:, 6], expected_normalized_error)
    assert jnp.isfinite(context).all()


def test_effectiveness_defaults_to_nominal_for_tiny_commands() -> None:
    commanded = jnp.array([[0.0, 1e-5, 2.0]], dtype=jnp.float32)
    applied = jnp.array([[0.4, 0.0, 1.0]], dtype=jnp.float32)
    eta = estimate_thruster_effectiveness(commanded, applied)

    assert jnp.allclose(eta, jnp.array([[1.0, 1.0, 0.5]], dtype=jnp.float32))


def test_replay_buffer_terminated_vs_truncated_bootstrap_returns() -> None:
    gamma = 0.9
    buf = ReplayBuffer(
        num_envs=1,
        obs_dim=1,
        act_dim=1,
        res_dim=0,
        size=2,
        gamma=gamma,
        lam=1.0,
    )

    obs = jnp.zeros((2, 1, 1), dtype=jnp.float32)
    act = jnp.zeros((2, 1, 1), dtype=jnp.float32)
    rew = jnp.array([[1.0], [1.0]], dtype=jnp.float32)
    val = jnp.zeros((2, 1), dtype=jnp.float32)
    logp = jnp.zeros((2, 1), dtype=jnp.float32)
    residuals = jnp.zeros((2, 1, 0), dtype=jnp.float32)
    buf.store_batch(obs, act, rew, val, logp, residuals)

    done_masks = jnp.array([[True], [True]])
    terminated_masks = jnp.array([[True], [False]])
    truncated_masks = jnp.array([[False], [True]])
    bootstrap = jnp.array([[0.0], [10.0]], dtype=jnp.float32)

    buf.finalize_with_masks(
        done_masks=done_masks,
        bootstrap_values=bootstrap,
        terminated_masks=terminated_masks,
        truncated_masks=truncated_masks,
    )
    data = buf.get()
    returns = data["ret"][:, 0]

    assert jnp.isclose(returns[0], 1.0)  # terminated: no bootstrap
    assert jnp.isclose(returns[1], 1.0 + gamma * 10.0)  # truncated: bootstrap
