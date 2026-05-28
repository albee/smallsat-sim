from smallsat_sim.controllers.rl.runners.curriculum import (
    build_authority_regime_curriculum,
    build_authority_regime_curriculum_v2,
    build_residual_policy_curriculum,
    phase_failure_fraction,
)


def test_authority_curriculum_has_single_mixed_failure_phase_after_warmup() -> None:
    phases, total_epochs = build_authority_regime_curriculum(
        train_with_failures=True,
        fallback_epochs=900,
        nominal_epochs=600,
        phase_epochs=200,
        failure_fraction=0.5,
        disturbance_fraction=0.1,
    )

    assert len(phases) == 2
    assert phases[0]["name"] == "warmup_nominal"
    assert phases[0]["epochs"] == 600
    assert phases[0]["failure_fraction"] == 0.0
    assert phases[1]["name"] == "mixed_failures"
    assert phases[1]["epochs"] == 200
    assert phases[1]["failure_fraction"] == 0.5
    assert total_epochs == 800

    mix = phases[1]["failure_sampling_mix"]
    assert [entry["name"] for entry in mix] == [
        "hard_feasible",
        "easy_feasible",
        "near_infeasible",
        "bias_or_nonlinear",
    ]
    assert [entry["weight"] for entry in mix] == [33, 7, 7, 3]


def test_authority_curriculum_v2_default_schedule() -> None:
    phases, total_epochs = build_authority_regime_curriculum_v2(
        train_with_failures=True,
        fallback_epochs=2000,
        nominal_epochs=100,
        phase_epochs=50,
        failure_fraction=0.5,
        disturbance_fraction=0.1,
    )

    assert len(phases) == 1
    assert phases[0]["name"] == "mixed_failure_ramp"
    assert phases[0]["epochs"] == 2000
    assert phase_failure_fraction(phases[0], 0) == 0.0
    assert phase_failure_fraction(phases[0], 750) == 0.25
    assert phase_failure_fraction(phases[0], 1500) == 0.5
    assert phase_failure_fraction(phases[0], 2000) == 0.5
    assert total_epochs == 2000

    mix = phases[0]["failure_sampling_mix"]
    assert [entry["name"] for entry in mix] == [
        "hard_feasible",
        "easy_feasible",
        "near_infeasible",
        "bias_or_nonlinear",
    ]
    assert [entry["weight"] for entry in mix] == [70, 15, 10, 5]


def test_authority_curriculum_v2_nominal_only_unchanged() -> None:
    phases, total_epochs = build_authority_regime_curriculum_v2(
        train_with_failures=False,
        fallback_epochs=900,
    )

    assert phases == [
        {
            "name": "nominal_only",
            "epochs": 900,
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
    assert total_epochs == 900


def test_residual_policy_curriculum_default_phases() -> None:
    phases, total_epochs = build_residual_policy_curriculum(
        train_with_failures=True,
        fallback_epochs=900,
    )
    assert total_epochs == 1000
    assert [phase["name"] for phase in phases] == [
        "residual_warmup",
        "residual_main",
        "residual_target",
    ]
    assert [phase["epochs"] for phase in phases] == [100, 200, 700]
    assert [phase["failure_fraction"] for phase in phases] == [0.20, 0.40, 0.50]
    assert "failure_fraction_schedule" not in phases[0]
    assert "failure_fraction_schedule" not in phases[1]
    assert "failure_fraction_schedule" not in phases[2]

    mix0 = phases[0]["failure_sampling_mix"]
    assert [entry["name"] for entry in mix0] == ["easy_feasible", "hard_feasible"]
    assert [entry["weight"] for entry in mix0] == [80, 20]

    mix1 = phases[1]["failure_sampling_mix"]
    assert [entry["name"] for entry in mix1] == ["easy_feasible", "hard_feasible"]
    assert [entry["weight"] for entry in mix1] == [30, 70]

    mix2 = phases[2]["failure_sampling_mix"]
    assert [entry["name"] for entry in mix2] == [
        "easy_feasible",
        "hard_feasible",
        "near_infeasible",
        "bias_or_nonlinear",
    ]
    assert [entry["weight"] for entry in mix2] == [15, 70, 10, 5]


def test_residual_policy_curriculum_nominal_only_unchanged() -> None:
    phases, total_epochs = build_residual_policy_curriculum(
        train_with_failures=False,
        fallback_epochs=123,
    )
    assert phases == [
        {
            "name": "nominal_only",
            "epochs": 123,
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
    assert total_epochs == 123
