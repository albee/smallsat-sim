from __future__ import annotations

import jax.numpy as jnp

# VecEnv distribution order:
# [STUCK_OFF, STUCK_ON, FAULTY_VALVE, SATURATED_THRUST, THRUST_INSTABILITY]
FAILURE_NAMES: dict[int, str] = {
    0: "STUCK_OFF",
    1: "STUCK_ON",
    2: "FAULTY_VALVE",
    3: "SATURATED_THRUST",
    4: "THRUST_INSTABILITY",
}

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


def build_failure_curriculum(
    *,
    train_with_failures: bool,
    fallback_epochs: int,
    nominal_epochs: int = 100,
    phase_epochs: int = 50,
    failure_fraction: float = 0.4,
    disturbance_fraction: float = 0.1,
) -> tuple[list[dict], int]:
    """
    Build the curriculum phases and return (phases, total_epochs).
    """
    if not train_with_failures:
        phases = [
            {
                "name": "nominal_only",
                "epochs": int(fallback_epochs),
                "active_failures": [],
                "failure_fraction": 0.0,
                "disturbance_fraction": 0.0,
                "new_failure": None,
            }
        ]
        return phases, int(fallback_epochs)

    phases: list[dict] = [
        {
            "name": "nominal",
            "epochs": int(nominal_epochs),
            "active_failures": [],
            "failure_fraction": 0.0,
            "disturbance_fraction": 0.0,
            "new_failure": None,
        }
    ]

    active: list[int] = []
    for failure_idx in FAILURE_ORDER:
        active.append(failure_idx)
        phases.append(
            {
                "name": FAILURE_NAMES[failure_idx],
                "epochs": int(phase_epochs),
                "active_failures": list(active),
                "failure_fraction": float(failure_fraction),
                "disturbance_fraction": 0.0,
                "new_failure": failure_idx,
            }
        )

    # Final phase: keep the failure mixture and add disturbances.
    phases.append(
        {
            "name": "disturbances",
            "epochs": int(phase_epochs),
            "active_failures": list(active),
            "failure_fraction": float(failure_fraction),
            "disturbance_fraction": float(disturbance_fraction),
            "new_failure": FAILURE_ORDER[-1],
        }
    )

    total_epochs = sum(int(phase["epochs"]) for phase in phases)
    return phases, total_epochs
