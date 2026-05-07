from types import SimpleNamespace

import jax
import jax.numpy as jnp

from smallsat_sim.envs.astrobee_rl.env import AstrobeeEnvVectorized
from smallsat_sim.envs.perturbations_rl import Perturbation, PerturbationStatus


def _make_dummy_env_cfg(num_envs: int):
    return SimpleNamespace(
        control=SimpleNamespace(RL=SimpleNamespace(num_envs=num_envs)),
        sim=SimpleNamespace(verbose=False),
    )


def _make_dummy_model_cfg(num_thrusters: int):
    thrusters = [
        SimpleNamespace(forcerange=(0.0, 1.0), ctrlrange=(0.0, 1.0))
        for _ in range(num_thrusters)
    ]
    return SimpleNamespace(
        Thrusters=SimpleNamespace(n_thrusters=num_thrusters, thruster_list=thrusters)
    )


def test_reset_perturbations_clears_shared_thruster_mask() -> None:
    num_envs = 3
    num_thrusters = 4

    # Simulate stale global failure state from a previous epoch/run.
    Perturbation.thruster_mask = jnp.full(
        (num_envs, num_thrusters),
        PerturbationStatus.STUCK_OFF.value,
        dtype=jnp.int32,
    )

    class _FakeEnv:
        def __init__(self):
            self.env_cfg = _make_dummy_env_cfg(num_envs)
            self.model_cfg = _make_dummy_model_cfg(num_thrusters)

        def next_rng_keys(self, count: int):
            return jax.random.split(jax.random.PRNGKey(0), count)

        def _refresh_effect_states(self) -> None:
            pass

    fake_env = _FakeEnv()

    AstrobeeEnvVectorized.reset_perturbations(fake_env)

    assert Perturbation.thruster_mask.shape == (num_envs, num_thrusters)
    assert jnp.all(Perturbation.thruster_mask == PerturbationStatus.OPERATIONAL.value)
