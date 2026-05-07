from types import SimpleNamespace

import jax
import jax.numpy as jnp

from smallsat_sim.envs.perturbations_rl import (
    FaultyValve,
    Perturbation,
    PerturbationStatus,
)


def _make_env_cfg(num_envs: int):
    return SimpleNamespace(
        control=SimpleNamespace(RL=SimpleNamespace(num_envs=num_envs)),
        sim=SimpleNamespace(verbose=False),
    )


def _make_model_cfg(num_thrusters: int):
    thruster_list = [
        SimpleNamespace(ctrlrange=(0.0, 1.0), forcerange=(0.0, 1.0))
        for _ in range(num_thrusters)
    ]
    return SimpleNamespace(
        Thrusters=SimpleNamespace(n_thrusters=num_thrusters, thruster_list=thruster_list)
    )


def test_gp_registration_skips_when_no_shared_operational_thruster() -> None:
    Perturbation.thruster_mask = None
    env_cfg = _make_env_cfg(num_envs=2)
    model_cfg = _make_model_cfg(num_thrusters=2)
    gp = FaultyValve(env_cfg, model_cfg, key=jax.random.PRNGKey(0))

    # Make env 0 only thruster 1 operational, env 1 only thruster 0 operational.
    Perturbation.thruster_mask = Perturbation.thruster_mask.at[0, 0].set(
        PerturbationStatus.STUCK_OFF.value
    )
    Perturbation.thruster_mask = Perturbation.thruster_mask.at[1, 1].set(
        PerturbationStatus.STUCK_ON.value
    )
    before = Perturbation.thruster_mask

    gp.register_perturbation(
        key=jax.random.PRNGKey(1),
        perturbed_envs=jnp.array([0, 1], dtype=jnp.int32),
    )

    assert jnp.array_equal(Perturbation.thruster_mask, before)
