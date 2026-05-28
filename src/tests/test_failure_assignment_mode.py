import os


def test_config_defaults_to_persistent_assignment_mode() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_path = os.path.join(
        repo_root, "src", "smallsat_sim", "envs", "astrobee_rl", "cfg", "config.py"
    )
    with open(config_path, encoding="utf-8") as file:
        src = file.read()

    assert 'curriculum_failure_assignment_mode = "persistent"' in src
    assert "curriculum_failure_env_count_quantum = 64" in src
    assert "curriculum_failure_assignment_refresh_fraction = 0.02" in src
    assert "curriculum_failure_assignment_refresh_interval = 10" in src


def test_legacy_mode_forces_quantum_one_in_training_loop() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    training_loop_path = os.path.join(
        repo_root,
        "src",
        "smallsat_sim",
        "controllers",
        "rl",
        "runners",
        "training_loop.py",
    )
    with open(training_loop_path, encoding="utf-8") as file:
        src = file.read()

    assert 'getattr(cfg, "curriculum_failure_assignment_mode", "persistent")' in src
    assert 'if assignment_mode not in ("persistent", "legacy"):' in src
    assert 'if assignment_mode == "legacy"' in src
    assert "f\"quantum={failure_env_count_quantum} \"" in src
