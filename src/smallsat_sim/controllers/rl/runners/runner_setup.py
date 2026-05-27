import os


def configure_jax_compilation_cache(jax_module) -> None:
    """
    Enable persistent JAX compilation cache when supported by the installed JAX.
    """
    if os.environ.get("SMALLSAT_DISABLE_JAX_CACHE", "0") == "1":
        return

    cache_dir = os.environ.get(
        "SMALLSAT_JAX_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "smallsat-sim", "jax"),
    )
    try:
        os.makedirs(cache_dir, exist_ok=True)
        jax_module.config.update("jax_compilation_cache_dir", cache_dir)
    except Exception:
        # Keep training functional on older JAX versions or read-only homes.
        pass


def build_checkpoint_file_names(env) -> dict[str, str]:
    """Create checkpoint/data filenames from the active RL configuration."""
    adaptive = "adaptive" if env.use_adaptive_approach else None
    context_mode = env.adaptive_context_mode if env.use_adaptive_approach else None
    context_version = (
        "authorityctx"
        if env.use_adaptive_approach and env.adaptive_context_mode == "structured"
        else None
    )
    pretrained = "pretrained" if env.use_pretrained else None
    nominal = None if env.train_with_failures else "nominal"
    scratch_failure = (
        "scratchfail" if env.train_with_failures and not env.use_pretrained else None
    )
    success_criterion = getattr(
        env.env_cfg.control.RL,
        "success_criterion",
        "full_pose",
    )
    task_tag = "full_pose" if success_criterion == "full_pose" else None
    curriculum_mode = str(
        getattr(env.env_cfg.control.RL, "failure_curriculum_mode", "type")
    )
    task_feasible_tag = (
        "taskaligned"
        if env.train_with_failures and curriculum_mode == "authority"
        else None
    )
    failure_fraction_tag = None
    if env.train_with_failures:
        failure_fraction = float(
            getattr(env.env_cfg.control.RL, "curriculum_failure_fraction", 0.0)
        )
        failure_fraction_tag = f"ff{int(round(100.0 * failure_fraction))}"

    def build_name(prefix: str) -> str:
        parts = [
            prefix,
            adaptive,
            context_mode,
            context_version,
            pretrained,
            task_tag,
            scratch_failure,
            task_feasible_tag,
            failure_fraction_tag,
            nominal,
        ]
        predictive_am = (
            env.use_task_conditioned_am
            or float(env.env_cfg.control.RL.am_predict_delta_weight) > 0.0
            or float(env.env_cfg.control.RL.am_predict_tracking_weight) > 0.0
            or float(getattr(env.env_cfg.control.RL, "am_predict_authority_weight", 0.0))
            > 0.0
        )
        if prefix.startswith("adapt_module") and predictive_am:
            parts.append("taskpred")
            if (
                float(
                    getattr(
                        env.env_cfg.control.RL,
                        "am_predict_authority_weight",
                        0.0,
                    )
                )
                > 0.0
            ):
                parts.append("authority")
        return "_".join(p for p in parts if p) + ".pkl"

    if adaptive:
        pretraining_data = build_name("pretraining_data")
        pretraining_state = build_name("pretraining_state")
    else:
        pretraining_data = "pretraining_data.pkl"
        pretraining_state = "pretraining_state.pkl"

    return {
        "pretraining_data_file_name": pretraining_data,
        "pretraining_state_file_name": pretraining_state,
        "training_data_file_name": build_name("training_data"),
        "training_state_file_name": build_name("training_state"),
        "adaptation_module_file_name": build_name(
            f"adapt_module_state_{env.am_architecture}"
        ),
    }
