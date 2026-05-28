import os

import numpy as np

from smallsat_sim.controllers.rl.runners import failure_scenarios as fs


def _fake_metrics_for_combo(failure_types):
    # Match _scenario_metrics return shape while keeping this test fast/stable.
    infeasible = int(failure_types[0]) == 1
    targeted_error = 0.40 if infeasible else 0.02
    targeted_margin = -0.40 if infeasible else 0.20
    return (
        6,  # rank
        1.0,  # min_sv
        10.0,  # condition_number
        0.01,  # mean_error
        0.02,  # p90_error
        0.0,  # bias_cancellation_error
        0.0,  # bias_wrench_norm
        0.2,  # p10_authority_margin
        0.3,  # mean_authority_margin
        fs.REGIME_MARGINAL,  # authority_regime
        targeted_error,
        targeted_margin,
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),  # targeted_task_wrench
        0.75,  # targeted_task_utilization
        0.80,  # horizon_max_utilization
        0.60,  # horizon_mean_utilization
        fs.LABEL_TORQUE_DEGENERATE,  # authority_label_mask
        np.zeros((6,), dtype=np.float32),  # disturbance_wrench
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),  # targeted_task_direction
    )


def test_include_infeasible_routes_to_stress_split(monkeypatch) -> None:
    combos = [((0, 0),), ((1, 1),)]

    def fake_iter_failure_combos(**_kwargs):
        return combos

    def fake_scenario_metrics(
        failure_types,
        _thrusters,
        _mixer_t,
        _ctrl_low,
        _ctrl_high,
        _task_wrenches,
        _targeted_task_wrenches,
        _mild_effectiveness,
    ):
        return _fake_metrics_for_combo(failure_types)

    monkeypatch.setattr(fs, "_iter_failure_combos", fake_iter_failure_combos)
    monkeypatch.setattr(fs, "_scenario_metrics", fake_scenario_metrics)

    mixer_t = np.zeros((2, 6), dtype=np.float32)
    ctrl_low = np.zeros((2,), dtype=np.float32)
    ctrl_high = np.ones((2,), dtype=np.float32)

    table_without = fs._build_scenario_table_cached(
        mixer_t.tobytes(),
        tuple(mixer_t.shape),
        ctrl_low.tobytes(),
        ctrl_high.tobytes(),
        1,
        1,
        0,
        1,
        0.9,
        0.5,
        False,
    )
    assert table_without["task_feasibility_regime"].shape[0] == 1
    assert np.all(table_without["task_feasibility_regime"] != fs.TASK_REGIME_INFEASIBLE)

    table_with = fs._build_scenario_table_cached(
        mixer_t.tobytes(),
        tuple(mixer_t.shape),
        ctrl_low.tobytes(),
        ctrl_high.tobytes(),
        1,
        1,
        0,
        1,
        0.9,
        0.5,
        True,
    )
    regimes = table_with["task_feasibility_regime"]
    splits = table_with["split"]
    infeasible_mask = regimes == fs.TASK_REGIME_INFEASIBLE
    assert int(infeasible_mask.sum()) == 1
    assert np.all(splits[infeasible_mask] == fs.SPLIT_STRESS_TEST)


def test_training_and_adaptation_are_authority_curriculum_only() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    training_loop_path = os.path.join(
        repo_root, "src", "smallsat_sim", "controllers", "rl", "runners", "training_loop.py"
    )
    adaptation_training_path = os.path.join(
        repo_root, "src", "smallsat_sim", "controllers", "rl", "runners", "adaptation_training.py"
    )

    with open(training_loop_path, encoding="utf-8") as file:
        training_src = file.read()
    with open(adaptation_training_path, encoding="utf-8") as file:
        adaptation_src = file.read()

    assert "build_authority_regime_curriculum_v2(" in training_src
    assert "build_authority_regime_curriculum_v2(" in adaptation_src
    assert "build_residual_policy_curriculum(" in training_src
    assert "build_residual_policy_curriculum(" in adaptation_src
    assert 'adaptive_policy_mode == "residual"' in training_src
    assert 'adaptive_policy_mode == "residual"' in adaptation_src
    assert "build_failure_curriculum(" not in training_src
    assert "build_failure_curriculum(" not in adaptation_src
    assert "build_difficulty_curriculum(" not in training_src
    assert "build_difficulty_curriculum(" not in adaptation_src


def test_training_resets_disturbances_when_resampling_failures() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    training_loop_path = os.path.join(
        repo_root, "src", "smallsat_sim", "controllers", "rl", "runners", "training_loop.py"
    )

    with open(training_loop_path, encoding="utf-8") as file:
        training_src = file.read()

    assert "if hasattr(self.env, \"reset_disturbances\"):" in training_src
    assert "apply_sampled_failure_scenario_split(" in training_src
