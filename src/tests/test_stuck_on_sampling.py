from types import SimpleNamespace

import jax
import jax.numpy as jnp

from smallsat_sim.envs.perturbations_rl import Perturbation, StuckOnThrusters


def _make_env_cfg(num_envs: int):
    return SimpleNamespace(
        control=SimpleNamespace(RL=SimpleNamespace(num_envs=num_envs)),
        sim=SimpleNamespace(verbose=False),
    )


def _make_model_cfg(forceranges: list[tuple[float, float]]):
    thruster_list = [SimpleNamespace(forcerange=fr) for fr in forceranges]
    return SimpleNamespace(
        Thrusters=SimpleNamespace(
            n_thrusters=len(forceranges),
            thruster_list=thruster_list,
        )
    )


def test_stuck_on_samples_uniform_force_within_thruster_range() -> None:
    Perturbation.thruster_mask = None
    forceranges = [(0.1, 1.0), (0.2, 0.9)]
    pert = StuckOnThrusters(
        _make_env_cfg(num_envs=2),
        _make_model_cfg(forceranges),
        key=jax.random.PRNGKey(0),
    )

    envs = jnp.array([0, 1], dtype=jnp.int32)
    thrusters = jnp.array([0, 1], dtype=jnp.int32)
    pert.stuck_on_thruster(
        key=jax.random.PRNGKey(1),
        perturbed_envs=envs,
        perturbed_thrusters=thrusters,
    )

    sampled = pert.state.max_thruster_force[envs, thrusters]
    mins = jnp.array([forceranges[0][0], forceranges[1][0]], dtype=sampled.dtype)
    maxs = jnp.array([forceranges[0][1], forceranges[1][1]], dtype=sampled.dtype)

    assert jnp.all(sampled >= mins)
    assert jnp.all(sampled < maxs)
