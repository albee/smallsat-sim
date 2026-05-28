import os

# Set flags to improve XLA performance on GPU
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "

from experiments.rl_benchmarking.rl_benchmarker import Benchmarker


benchmarker = Benchmarker(run_name="ppo_adaptive_transformer_residual")
nominal_ckpt = os.environ.get("SMALLSAT_NOMINAL_CKPT_ALIAS", "nominal_actor_baseline.pkl")

benchmarker.train_and_evaluate(
    train_with_failures=True,
    use_pretrained=False,
    use_adaptive_approach=True,
    am_architecture="transformer",
    adaptive_context_mode="residual",
    adaptive_policy_mode="residual",
    context_fusion="film",
    residual_scale=0.25,
    frozen_nominal_actor=True,
    nominal_actor_checkpoint=nominal_ckpt,
)

benchmarker.deploy_and_test(
    train_with_failures=True,
    use_pretrained=False,
    use_adaptive_approach=True,
    am_architecture="transformer",
    adaptive_context_mode="residual",
    adaptive_policy_mode="residual",
    context_fusion="film",
    residual_scale=0.25,
    frozen_nominal_actor=True,
    nominal_actor_checkpoint=nominal_ckpt,
)
