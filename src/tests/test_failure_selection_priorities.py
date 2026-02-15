from dataclasses import dataclass

import jax
import jax.numpy as jnp

from smallsat_sim.envs.disturbances import DisturbanceState, DisturbanceStatus
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

    def _record(self, envs: jnp.ndarray) -> None:
        envs = jnp.asarray(envs, dtype=jnp.int32)
        self.calls.append(envs)
        if envs.size > 0:
            Perturbation.thruster_mask = Perturbation.thruster_mask.at[
                envs, self.thruster_idx
            ].set(self.failure_value)

    def stuck_off_thruster(self, _key, envs):
        self._record(envs)

    def stuck_on_thruster(self, _key, envs):
        self._record(envs)

    def register_perturbation(self, _key, envs):
        self._record(envs)


class _DisturbanceRecorder:
    def __init__(self) -> None:
        self.failure_type = DisturbanceStatus.CONSTANT_FORCE
        self.calls: list[jnp.ndarray] = []

    def const_force_disturbance(self, envs):
        self.calls.append(jnp.asarray(envs, dtype=jnp.int32))


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
