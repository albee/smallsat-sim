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


def phase_failure_fraction(phase: dict, curriculum_epoch: int) -> float:
    """Return the failure fraction for a phase at a zero-based curriculum epoch."""
    schedule = phase.get("failure_fraction_schedule")
    if not schedule:
        return float(phase["failure_fraction"])
    if schedule.get("type") != "linear_cap":
        raise ValueError(f"Unsupported failure_fraction_schedule: {schedule}")
    max_fraction = float(schedule["max_fraction"])
    ramp_epochs = max(1, int(schedule["ramp_epochs"]))
    return min(max_fraction, max_fraction * float(curriculum_epoch) / float(ramp_epochs))


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


def build_authority_regime_curriculum_v2(
    *,
    train_with_failures: bool,
    fallback_epochs: int,
    nominal_epochs: int = 200,
    phase_epochs: int = 200,
    failure_fraction: float = 0.5,
    disturbance_fraction: float = 0.1,
) -> tuple[list[dict], int]:
    """
    Default single-phase curriculum with immediate failure exposure.

    The total failed-env fraction follows
    ``min(0.5, 0.5 * epoch / 1500)`` for zero-based curriculum epochs. Failed
    environments always use the same scenario-regime mix:
    hard_feasible=70, easy_feasible=15, near_infeasible=10,
    bias_or_nonlinear=5. Non-failed environments remain nominal.
    """
    del nominal_epochs, phase_epochs, failure_fraction, disturbance_fraction
    if not train_with_failures:
        phases = [
            {
                "name": "nominal_only",
                "epochs": 1000,
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
        return phases, 1000

    total_epochs = max(int(fallback_epochs), 1501)
    phases: list[dict] = [
        {
            "name": "mixed_failure_ramp",
            "epochs": total_epochs,
            "active_failures": list(FAILURE_ORDER),
            "failure_fraction": 0.0,
            "failure_fraction_schedule": {
                "type": "linear_cap",
                "max_fraction": 0.5,
                "ramp_epochs": 1500,
            },
            "disturbance_fraction": 0.0,
            "new_failure": None,
            "difficulty_bin": None,
            "authority_regime": None,
            "task_feasibility_regime": None,
            "authority_label_any_mask": None,
            "failure_sampling_mix": (
                {
                    "name": "hard_feasible",
                    "weight": 70,
                    "task_feasibility_regime": TASK_REGIME_HARD_FEASIBLE,
                    "authority_label_any_mask": None,
                },
                {
                    "name": "easy_feasible",
                    "weight": 15,
                    "task_feasibility_regime": TASK_REGIME_EASY_FEASIBLE,
                    "authority_label_any_mask": None,
                },
                {
                    "name": "near_infeasible",
                    "weight": 10,
                    "task_feasibility_regime": TASK_REGIME_NEAR_INFEASIBLE,
                    "authority_label_any_mask": None,
                },
                {
                    "name": "bias_or_nonlinear",
                    "weight": 5,
                    "task_feasibility_regime": None,
                    "authority_label_any_mask": int(
                        LABEL_BIAS_DOMINATED | LABEL_NONLINEAR_MISMATCH
                    ),
                },
            ),
        },
    ]
    return phases, total_epochs
