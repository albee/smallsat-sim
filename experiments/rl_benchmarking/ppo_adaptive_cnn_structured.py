import os

# Set flags to improve XLA performance on GPU
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "

from experiments.rl_benchmarking.rl_benchmarker import Benchmarker


benchmarker = Benchmarker(run_name="ppo_adaptive_cnn_structured")

benchmarker.train_and_evaluate(
    train_with_failures=True,
    use_pretrained=False,
    use_adaptive_approach=True,
    am_architecture="cnn",
    adaptive_context_mode="structured",
)

benchmarker.deploy_and_test(
    train_with_failures=True,
    use_pretrained=False,
    use_adaptive_approach=True,
    am_architecture="cnn",
    adaptive_context_mode="structured",
)
