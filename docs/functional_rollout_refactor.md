# Functional Rollout Refactor Overview

This document captures the functional‐rollout work that landed in this branch so future contributors can understand the design and the testing hooks that were added.

## Goals

- Replace the imperative per-step environment loop with a pure JAX implementation so PPO rollouts can be `lax.scan`’d and JIT compiled.
- Ensure the refactor stays behaviourally equivalent to the legacy path.
- Leave existing deployment/benchmarking scripts working (they still use the imperative controller APIs).

## Key changes

| Area | Description |
| --- | --- |
| `VecEnv` | Added `vecenv_step` and related helpers that operate purely on `VecEnvState`. MuJoCo stepping is now batched via `jax.vmap`, and a `VecEnvStepOutput` object carries rewards, wrench data, observations, etc. |
| `Perturbations` | Converted stuck/faulty helpers to be JAX-friendly (no Python boolean branches or boolean indexing on traced arrays). |
| `OnPolicyRunner` | New `run_functional_rollout` orchestrates the scan. We register the internal dataclasses as pytrees, added optional regression checking (`verify_functional_rollout`), and updated the logging paths. |
| `RLController` | PD / deployment scripts explicitly call the legacy `VecEnv.transition` to avoid mixing functional state snapshots with imperative MuJoCo calls. Checkpoint loading now supports both dict states and legacy module wrappers. |
| `Logger` | Keys now include `(stage, step)` so multiple stages do not overwrite each other, keeping plots readable after the refactor. |

## Verification hooks

- Set `control.RL.verify_functional_rollout = True` to check one rollout step against the old imperative transition. Training will raise immediately if a mismatch appears.
- Logging still writes to `logs/<timestamp>`; run `rl_plotter.py` to produce the same plots as before the refactor.

## Things to keep in mind

- Functional rollouts are only used during RL training / AM training / evaluation. Deployment scripts (and pretraining) still rely on `RLController.control`, which routes through the legacy environment interface.
- When adding new data to `VecEnvStepOutput` or the extra carry, register any new dataclasses with `jax.tree_util.register_pytree_node`.
- The perturbation helpers now expect per-env/per-thruster arrays and broadcast them via `_broadcast_to_control_shape`; keep future perturbation state snapshots consistent with that convention.

Feel free to enable the verification flag when modifying the functional path; it adds negligible overhead for short runs and quickly catches behavioural regressions.

