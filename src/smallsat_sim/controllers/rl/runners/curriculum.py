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

    After the clean nominal phase, fine-tuning uses a 50/50 mixture by default:
    clean environments and task-aligned hard-feasible failures. In the failed
    environments, the task wrench is certified feasible but near the damaged
    actuator boundary.
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

    weighted_mix = (
        # 50% clean / nominal
        ("nominal", 50, None, None),
        # easy_feasible small amount only
        ("easy_feasible", 2, TASK_REGIME_EASY_FEASIBLE, None),
        # 35% hard_feasible task-conditioned (main target)
        ("hard_feasible", 33, TASK_REGIME_HARD_FEASIBLE, None),
        # 10% near_infeasible task-conditioned (smaller amount)
        ("near_infeasible", 10, TASK_REGIME_NEAR_INFEASIBLE, None),
        # 5% bias/nonlinear cases
        (
            "bias_or_nonlinear",
            5,
            None,
            int(LABEL_BIAS_DOMINATED | LABEL_NONLINEAR_MISMATCH),
        ),
    )

    total_weight = sum(weight for _, weight, _, _ in weighted_mix)
    scaled_epochs = [
        max(1, int(round(float(phase_epochs) * float(weight) / float(total_weight))))
        for _, weight, _, _ in weighted_mix
    ]
    diff = int(phase_epochs) - int(sum(scaled_epochs))
    scaled_epochs[0] += diff

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

    for (name, _weight, regime, label_mask), epochs in zip(weighted_mix, scaled_epochs):
        is_nominal = name == "nominal"
        phases.append(
            {
                "name": name,
                "epochs": int(epochs),
                "active_failures": [] if is_nominal else list(FAILURE_ORDER),
                "failure_fraction": 0.0 if is_nominal else 1.0,
                "disturbance_fraction": 0.0,
                "new_failure": None,
                "difficulty_bin": None,
                "authority_regime": None,
                "task_feasibility_regime": regime,
                "authority_label_any_mask": label_mask,
            }
        )

    total_epochs = sum(int(phase["epochs"]) for phase in phases)
    return phases, total_epochs
