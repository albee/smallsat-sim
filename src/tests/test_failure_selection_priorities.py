from dataclasses import dataclass

import jax
import jax.numpy as jnp

from smallsat_sim.envs.disturbances import DisturbanceState, DisturbanceStatus
from smallsat_sim.controllers.rl.runners.failure_scenarios import (
    SPLIT_TRAIN,
    TASK_REGIME_HARD_FEASIBLE,
    _iter_failure_combos,
    sample_task_conditioned_scenario_indices,
    task_wrench_from_state_features,
)
from smallsat_sim.envs.perturbations_rl import Perturbation, PerturbationStatus
from smallsat_sim.envs.vec_env import VecEnv


@dataclass
class _DummyState:
    disturbance_states: tuple = ()
    perturbation_states: tuple = ()

    def replace(self, **updates):
        for key, value in updates.items():
            setattr(self, key, value)
        return self


class _PerturbationRecorder:
    def __init__(self, failure_value: int, thruster_idx: int = 0) -> None:
        self.failure_value = int(failure_value)
        self.thruster_idx = int(thruster_idx)
        self.calls: list[jnp.ndarray] = []
        self.start_times: list[float | None] = []

    def _record(self, envs: jnp.ndarray, start_time: float | None) -> None:
        envs = jnp.asarray(envs, dtype=jnp.int32)
        self.calls.append(envs)
        self.start_times.append(start_time)
        if envs.size > 0:
            Perturbation.thruster_mask = Perturbation.thruster_mask.at[
                envs, self.thruster_idx
            ].set(self.failure_value)

    def stuck_off_thruster(self, _key, envs, start_time=None):
        self._record(envs, start_time)

    def stuck_on_thruster(self, _key, envs, start_time=None):
        self._record(envs, start_time)

    def register_perturbation(self, _key, envs, start_time=None):
        self._record(envs, start_time)


class _DisturbanceRecorder:
    def __init__(self) -> None:
        self.failure_type = DisturbanceStatus.CONSTANT_FORCE
        self.calls: list[jnp.ndarray] = []
        self.start_times: list[float] = []

    def const_force_disturbance(self, envs, start_time=0.0):
        self.calls.append(jnp.asarray(envs, dtype=jnp.int32))
        self.start_times.append(float(start_time))


def _make_minimal_vec_env(num_envs: int, num_thrusters: int) -> VecEnv:
    env = VecEnv.__new__(VecEnv)
    env.num_envs = int(num_envs)
    env.disturbance_states = ()
    env.perturbation_states = ()
    env._state = _DummyState()
    env._refresh_effect_states = lambda: None
    Perturbation.thruster_mask = jnp.full(
        (num_envs, num_thrusters), PerturbationStatus.OPERATIONAL.value, dtype=jnp.int32
    )
    return env


def test_random_perturbations_prioritize_clean_envs() -> None:
    env = _make_minimal_vec_env(num_envs=10, num_thrusters=2)

    # Existing failures on envs 0,1 and existing disturbances on envs 2,3.
    Perturbation.thruster_mask = Perturbation.thruster_mask.at[[0, 1], 0].set(
        PerturbationStatus.STUCK_OFF.value
    )
    env.disturbance_states = (
        DisturbanceState(
            rng=jax.random.PRNGKey(0),
            active_mask=jnp.array(
                [False, False, True, True, False, False, False, False, False, False]
            ),
            start_times=jnp.zeros((10,)),
            params={"const_force": jnp.zeros((10, 6))},
        ),
    )

    recorders = [
        _PerturbationRecorder(PerturbationStatus.STUCK_OFF.value),
        _PerturbationRecorder(PerturbationStatus.STUCK_ON.value),
        _PerturbationRecorder(PerturbationStatus.FAULTY_VALVE.value),
        _PerturbationRecorder(PerturbationStatus.SATURATED_THRUST.value),
        _PerturbationRecorder(PerturbationStatus.THRUST_INSTABILITY.value),
    ]
    env.perturbations = type("P", (), {"perturbations": recorders})()

    VecEnv.apply_random_perturbations(
        env,
        key=jax.random.PRNGKey(123),
        fraction_perturbed_envs=0.4,  # 4 envs
        perturbation_distribution=jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]),
    )

    picked = jnp.concatenate(
        [call for rec in recorders for call in rec.calls if call.size > 0]
    )
    assert picked.shape[0] == 4
    assert jnp.unique(picked).shape[0] == picked.shape[0]
    # Clean env pool is [4,5,6,7,8,9]; should be selected first.
    assert jnp.all(picked >= 4)


def test_training_eval_style_sampling_avoids_perturbation_disturbance_overlap() -> None:
    env = _make_minimal_vec_env(num_envs=20, num_thrusters=2)
    env.disturbance_states = (
        DisturbanceState(
            rng=jax.random.PRNGKey(0),
            active_mask=jnp.zeros((20,), dtype=bool),
            start_times=jnp.zeros((20,)),
            params={"const_force": jnp.zeros((20, 6))},
        ),
    )

    recorders = [
        _PerturbationRecorder(PerturbationStatus.STUCK_OFF.value),
        _PerturbationRecorder(PerturbationStatus.STUCK_ON.value),
        _PerturbationRecorder(PerturbationStatus.FAULTY_VALVE.value),
        _PerturbationRecorder(PerturbationStatus.SATURATED_THRUST.value),
        _PerturbationRecorder(PerturbationStatus.THRUST_INSTABILITY.value),
    ]
    env.perturbations = type("P", (), {"perturbations": recorders})()

    disturbance_recorder = _DisturbanceRecorder()
    env.disturbances = type("D", (), {"disturbances": [disturbance_recorder]})()

    VecEnv.apply_random_perturbations(
        env,
        key=jax.random.PRNGKey(10),
        fraction_perturbed_envs=0.4,  # 8 envs
        perturbation_distribution=jnp.array([0.2, 0.2, 0.2, 0.2, 0.2]),
    )
    perturbed_envs = jnp.where(
        jnp.any(Perturbation.thruster_mask != PerturbationStatus.OPERATIONAL.value, axis=1)
    )[0]
    assert perturbed_envs.shape[0] == 8

    VecEnv.apply_random_disturbance(
        env,
        key=jax.random.PRNGKey(11),
        fraction_disturbed_envs=0.1,  # 2 envs
    )
    disturbed_envs = disturbance_recorder.calls[-1]
    assert disturbed_envs.shape[0] == 2
    # With 40% perturbed and 10% disturbed, clean envs are sufficient;
    # selected disturbed envs should not overlap with perturbed envs.
    assert jnp.intersect1d(perturbed_envs, disturbed_envs).size == 0


def test_random_perturbations_apply_at_most_one_perturbation_per_env() -> None:
    env = _make_minimal_vec_env(num_envs=20, num_thrusters=4)
    recorders = [
        _PerturbationRecorder(PerturbationStatus.STUCK_OFF.value, thruster_idx=0),
        _PerturbationRecorder(PerturbationStatus.STUCK_ON.value, thruster_idx=1),
        _PerturbationRecorder(PerturbationStatus.FAULTY_VALVE.value, thruster_idx=2),
        _PerturbationRecorder(PerturbationStatus.SATURATED_THRUST.value, thruster_idx=3),
        _PerturbationRecorder(PerturbationStatus.THRUST_INSTABILITY.value, thruster_idx=0),
    ]
    env.perturbations = type("P", (), {"perturbations": recorders})()

    VecEnv.apply_random_perturbations(
        env,
        key=jax.random.PRNGKey(77),
        fraction_perturbed_envs=0.5,  # 10 envs
        perturbation_distribution=jnp.array([0.2, 0.2, 0.2, 0.2, 0.2]),
    )

    failed_counts = jnp.sum(
        Perturbation.thruster_mask != PerturbationStatus.OPERATIONAL.value, axis=1
    )
    failed_envs = jnp.where(failed_counts > 0)[0]
    assert failed_envs.shape[0] == 10
    assert int(failed_counts.max()) <= 1


def test_random_failures_forward_start_times() -> None:
    env = _make_minimal_vec_env(num_envs=10, num_thrusters=2)
    recorders = [
        _PerturbationRecorder(PerturbationStatus.STUCK_OFF.value),
        _PerturbationRecorder(PerturbationStatus.STUCK_ON.value),
        _PerturbationRecorder(PerturbationStatus.FAULTY_VALVE.value),
        _PerturbationRecorder(PerturbationStatus.SATURATED_THRUST.value),
        _PerturbationRecorder(PerturbationStatus.THRUST_INSTABILITY.value),
    ]
    env.perturbations = type("P", (), {"perturbations": recorders})()

    VecEnv.apply_random_perturbations(
        env,
        key=jax.random.PRNGKey(88),
        fraction_perturbed_envs=0.4,
        perturbation_distribution=jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        start_time=12.5,
    )

    assert recorders[0].start_times[-1] == 12.5

    disturbance_recorder = _DisturbanceRecorder()
    env.disturbances = type("D", (), {"disturbances": [disturbance_recorder]})()
    VecEnv.apply_random_disturbance(
        env,
        key=jax.random.PRNGKey(89),
        fraction_disturbed_envs=0.2,
        start_time=7.25,
    )

    assert disturbance_recorder.start_times[-1] == 7.25


def test_task_wrench_from_state_uses_full_pose_and_rates() -> None:
    states = jnp.array(
        [[1.0, -2.0, 0.5, 0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9]],
        dtype=jnp.float32,
    )

    wrench = task_wrench_from_state_features(
        states,
        kp_pos=2.0,
        kd_pos=3.0,
        kp_att=5.0,
        kd_att=7.0,
    )

    expected = jnp.array(
        [[-3.2, 5.5, -2.8, -4.4, 4.6, -4.8]],
        dtype=jnp.float32,
    )
    assert jnp.allclose(wrench, expected)


def test_task_conditioned_scenario_sampling_prefers_aligned_weak_direction() -> None:
    table = {
        "split": jnp.array([SPLIT_TRAIN, SPLIT_TRAIN], dtype=jnp.int32),
        "difficulty_bin": jnp.array([2, 2], dtype=jnp.int32),
        "authority_regime": jnp.array([1, 1], dtype=jnp.int32),
        "task_feasibility_regime": jnp.array(
            [TASK_REGIME_HARD_FEASIBLE, TASK_REGIME_HARD_FEASIBLE],
            dtype=jnp.int32,
        ),
        "targeted_task_wrench": jnp.array(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=jnp.float32,
        ),
        "targeted_task_direction": jnp.array(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=jnp.float32,
        ),
    }
    task_wrenches = jnp.tile(
        jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
        (512, 1),
    )

    sampled = sample_task_conditioned_scenario_indices(
        jax.random.PRNGKey(123),
        table,
        split_id=SPLIT_TRAIN,
        task_wrenches=task_wrenches,
        alignment_temperature=12.0,
    )

    assert float(jnp.mean(sampled == 0)) > 0.95


def test_failure_library_candidate_generation_includes_higher_order_faults() -> None:
    combos = _iter_failure_combos(
        n_thrusters=12,
        max_faults=12,
        exhaustive_faults=2,
        sampled_per_fault_count=8,
    )
    n_faults = [len(combo) for combo in combos]

    assert max(n_faults) > 2
    assert 12 in n_faults
    assert all(len({thruster for _, thruster in combo}) == len(combo) for combo in combos)
