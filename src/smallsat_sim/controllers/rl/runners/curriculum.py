from __future__ import annotations

import jax.numpy as jnp

from smallsat_sim.controllers.rl.runners.failure_scenarios import (
    LABEL_BIAS_DOMINATED,
    LABEL_NONLINEAR_MISMATCH,
    TASK_REGIME_EASY_FEASIBLE,
    TASK_REGIME_HARD_FEASIBLE,
    TASK_REGIME_NEAR_INFEASIBLE,
)

# Sequential order requested by the curriculum.
FAILURE_ORDER: list[int] = [0, 2, 3, 4, 1]


def uniform_failure_distribution(active_failure_indices: list[int]) -> jnp.ndarray:
    """Build a uniform distribution over the active failures."""
    dist = jnp.zeros((5,), dtype=jnp.float32)
    if not active_failure_indices:
        return dist
    weight = 1.0 / float(len(active_failure_indices))
    for idx in active_failure_indices:
        dist = dist.at[idx].set(weight)
    return dist


def build_authority_regime_curriculum(
    *,
    train_with_failures: bool,
    fallback_epochs: int,
    nominal_epochs: int = 100,
    phase_epochs: int = 50,
    failure_fraction: float = 0.5,
    disturbance_fraction: float = 0.1,
) -> tuple[list[dict], int]:
    """
    Build the main task-relevant authority curriculum.

    After the clean nominal warmup, immediately train with a fixed mixed batch:
    nominal environments plus failure environments every epoch. Failure sampling
    is biased toward task-relevant hard-feasible cases, with smaller fractions
    for easy/near-infeasible/bias regimes.
    """
    del disturbance_fraction
    if not train_with_failures:
        phases = [
            {
                "name": "nominal_only",
                "epochs": int(fallback_epochs),
                "active_failures": [],
                "failure_fraction": 0.0,
                "disturbance_fraction": 0.0,
                "new_failure": None,
                "difficulty_bin": None,
                "authority_regime": None,
                "task_feasibility_regime": None,
                "authority_label_any_mask": None,
            }
        ]
        return phases, int(fallback_epochs)

    failure_sampling_mix = (
        {
            "name": "hard_feasible",
            "weight": 33,
            "task_feasibility_regime": TASK_REGIME_HARD_FEASIBLE,
            "authority_label_any_mask": None,
        },
        {
            "name": "easy_feasible",
            "weight": 7,
            "task_feasibility_regime": TASK_REGIME_EASY_FEASIBLE,
            "authority_label_any_mask": None,
        },
        {
            "name": "near_infeasible",
            "weight": 7,
            "task_feasibility_regime": TASK_REGIME_NEAR_INFEASIBLE,
            "authority_label_any_mask": None,
        },
        {
            "name": "bias_or_nonlinear",
            "weight": 3,
            "task_feasibility_regime": None,
            "authority_label_any_mask": int(
                LABEL_BIAS_DOMINATED | LABEL_NONLINEAR_MISMATCH
            ),
        },
    )

    phases: list[dict] = []
    if int(nominal_epochs) > 0:
        phases.append(
            {
                "name": "warmup_nominal",
                "epochs": int(nominal_epochs),
                "active_failures": [],
                "failure_fraction": 0.0,
                "disturbance_fraction": 0.0,
                "new_failure": None,
                "difficulty_bin": None,
                "authority_regime": None,
            }
        )

    if int(phase_epochs) > 0:
        phases.append(
            {
                "name": "mixed_failures",
                "epochs": int(phase_epochs),
                "active_failures": list(FAILURE_ORDER),
                "failure_fraction": float(failure_fraction),
                "disturbance_fraction": 0.0,
                "new_failure": None,
                "difficulty_bin": None,
                "authority_regime": None,
                "task_feasibility_regime": None,
                "authority_label_any_mask": None,
                "failure_sampling_mix": failure_sampling_mix,
            }
        )

    total_epochs = sum(int(phase["epochs"]) for phase in phases)
    return phases, total_epochs
