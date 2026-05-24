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
    pretrained = "pretrained" if env.use_pretrained else None
    nominal = None if env.train_with_failures else "nominal"

    def build_name(prefix: str) -> str:
        parts = [prefix, adaptive, context_mode, pretrained, nominal]
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
