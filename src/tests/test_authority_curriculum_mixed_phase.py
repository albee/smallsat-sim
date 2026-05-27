from smallsat_sim.controllers.rl.runners.curriculum import (
    build_authority_regime_curriculum,
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

