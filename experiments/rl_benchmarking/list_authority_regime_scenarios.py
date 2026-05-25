"""Print representative failure scenarios from the authority-regime table."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from smallsat_sim.envs.astrobee_rl.cfg import config as env_config
from smallsat_sim.envs.dynamics import SymbolicModel
from smallsat_sim.model.astrobee.cfg import config as model_config
from smallsat_sim.controllers.rl.runners.failure_scenarios import (
    AUTHORITY_REGIME_NAMES,
    FAILURE_TYPE_NAMES,
    REGIME_AUTHORITY_LIMITED,
    REGIME_BIAS_LIMITED,
    REGIME_MARGINAL,
    REGIME_REDUNDANT,
    SPLIT_NAMES,
    SPLIT_TRAIN,
    TASK_REGIME_NAMES,
    TASK_REGIME_EASY_FEASIBLE,
    TASK_REGIME_HARD_FEASIBLE,
    TASK_REGIME_INFEASIBLE,
    TASK_REGIME_NEAR_INFEASIBLE,
    build_failure_scenario_table,
    scenario_authority_regime_counts,
    scenario_task_regime_counts,
)


REGIME_BY_NAME = {
    "redundant": REGIME_REDUNDANT,
    "marginal": REGIME_MARGINAL,
    "authority_limited": REGIME_AUTHORITY_LIMITED,
    "bias_limited": REGIME_BIAS_LIMITED,
}

TASK_REGIME_BY_NAME = {
    "any": None,
    "easy_feasible": TASK_REGIME_EASY_FEASIBLE,
    "hard_feasible": TASK_REGIME_HARD_FEASIBLE,
    "near_infeasible": TASK_REGIME_NEAR_INFEASIBLE,
    "infeasible": TASK_REGIME_INFEASIBLE,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--regime",
        choices=tuple(REGIME_BY_NAME),
        default="authority_limited",
    )
    parser.add_argument(
        "--task-regime",
        choices=tuple(TASK_REGIME_BY_NAME),
        default="any",
    )
    parser.add_argument("--split", type=int, default=SPLIT_TRAIN)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--sparse-active-thrusters", type=int, default=None)
    parser.add_argument("--sparse-max-active-sets", type=int, default=None)
    parser.add_argument(
        "--sort-by",
        choices=(
            "p10_authority_margin",
            "p90_feasibility_error",
            "bias_cancellation_error",
            "mean_authority_margin",
        ),
        default="p10_authority_margin",
    )
    parser.add_argument("--descending", action="store_true")
    return parser.parse_args()


def _failure_label(failure_type: int, thruster: int) -> str:
    name = FAILURE_TYPE_NAMES.get(int(failure_type), "none")
    return f"{name}@thruster{int(thruster)}"


def main() -> None:
    args = _parse_args()
    env_cfg = env_config.EnvConfig()
    model_cfg = model_config.ModelConfig()
    symbolic_model = SymbolicModel(model_cfg)
    mixer_t = jnp.asarray(symbolic_model.mixer.T, dtype=jnp.float32)
    thruster_ranges = [
        thruster.forcerange for thruster in model_cfg.Thrusters.thruster_list
    ]
    act_low = jnp.asarray([fr[0] for fr in thruster_ranges], dtype=jnp.float32)
    act_high = jnp.asarray([fr[1] for fr in thruster_ranges], dtype=jnp.float32)

    cfg = env_cfg.control.RL
    table = build_failure_scenario_table(
        mixer_t,
        act_low,
        act_high,
        max_faults=int(getattr(cfg, "failure_scenario_max_faults", 2)),
        min_rank=int(getattr(cfg, "failure_scenario_min_rank", 6)),
        stress_quantile=float(getattr(cfg, "failure_scenario_stress_quantile", 0.9)),
        mild_effectiveness=float(
            getattr(cfg, "failure_scenario_mild_effectiveness", 0.5)
        ),
        sparse_active_thrusters=(
            args.sparse_active_thrusters
            if args.sparse_active_thrusters is not None
            else getattr(cfg, "failure_scenario_sparse_active_thrusters", None)
        ),
        sparse_max_active_sets=(
            args.sparse_max_active_sets
            if args.sparse_max_active_sets is not None
            else int(getattr(cfg, "failure_scenario_sparse_max_active_sets", 32))
        ),
    )

    arrays = {key: np.asarray(jax.device_get(value)) for key, value in table.items()}
    regime_id = REGIME_BY_NAME[args.regime]
    mask = np.logical_and(
        arrays["split"] == int(args.split),
        arrays["authority_regime"] == int(regime_id),
    )
    task_regime_id = TASK_REGIME_BY_NAME[args.task_regime]
    if task_regime_id is not None:
        mask = np.logical_and(
            mask,
            arrays["task_feasibility_regime"] == int(task_regime_id),
        )
    indices = np.nonzero(mask)[0]
    order = np.argsort(arrays[args.sort_by][indices])
    if args.descending:
        order = order[::-1]
    indices = indices[order][: max(0, int(args.limit))]

    print("[Authority Scenario Table]")
    print(f"split counts by regime: {scenario_authority_regime_counts(table, args.split)}")
    print(
        f"split counts by task feasibility: "
        f"{scenario_task_regime_counts(table, args.split)}"
    )
    print(
        f"selected regime={args.regime} split={SPLIT_NAMES.get(args.split, args.split)} "
        f"task_regime={args.task_regime} count={int(mask.sum())} "
        f"showing={len(indices)} sort_by={args.sort_by}"
    )
    print()

    for scenario_id in indices:
        failure_types = arrays["failure_types"][scenario_id]
        thrusters = arrays["thrusters"][scenario_id]
        failures = [
            _failure_label(ft, th)
            for ft, th in zip(failure_types, thrusters, strict=True)
            if ft >= 0 and th >= 0
        ]
        targeted_wrench = arrays["targeted_task_wrench"][scenario_id]
        print(
            f"id={scenario_id:04d} "
            f"regime={AUTHORITY_REGIME_NAMES[int(arrays['authority_regime'][scenario_id])]} "
            f"task_regime={TASK_REGIME_NAMES[int(arrays['task_feasibility_regime'][scenario_id])]} "
            f"split={SPLIT_NAMES[int(arrays['split'][scenario_id])]} "
            f"rank={int(arrays['rank'][scenario_id])} "
            f"min_sv={arrays['min_singular_value'][scenario_id]:.4f} "
            f"cond={arrays['condition_number'][scenario_id]:.2f} "
            f"p10_margin={arrays['p10_authority_margin'][scenario_id]:.3f} "
            f"mean_margin={arrays['mean_authority_margin'][scenario_id]:.3f} "
            f"p90_err={arrays['p90_feasibility_error'][scenario_id]:.3f} "
            f"target_err={arrays['targeted_task_error'][scenario_id]:.3f} "
            f"target_margin={arrays['targeted_task_margin'][scenario_id]:.3f} "
            f"target_force={np.array2string(targeted_wrench[:3], precision=3)} "
            f"target_torque={np.array2string(targeted_wrench[3:], precision=3)} "
            f"bias_err={arrays['bias_cancellation_error'][scenario_id]:.3f} "
            f"bias_norm={arrays['bias_wrench_norm'][scenario_id]:.3f} "
            f"failures={', '.join(failures)}"
        )


if __name__ == "__main__":
    main()
