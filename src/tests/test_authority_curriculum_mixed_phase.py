from smallsat_sim.controllers.rl.runners.curriculum import (
    build_authority_regime_curriculum,
    build_authority_regime_curriculum_v2,
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
        fallback_epochs=100,
        nominal_epochs=100,
        phase_epochs=50,
        failure_fraction=0.5,
        disturbance_fraction=0.1,
    )

    assert len(phases) == 5
    assert [p["epochs"] for p in phases] == [200, 200, 200, 200, 200]
    assert [p["failure_fraction"] for p in phases] == [0.01, 0.05, 0.15, 0.30, 0.50]
    assert total_epochs == 1000

    last_mix = phases[-1]["failure_sampling_mix"]
    assert [entry["name"] for entry in last_mix] == [
        "easy_feasible",
        "hard_feasible",
        "near_infeasible",
        "bias_or_nonlinear",
    ]
    assert [entry["weight"] for entry in last_mix] == [7, 33, 7, 3]
