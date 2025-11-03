import os

# Set flags to improve XLA performance on GPU
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "

from experiments.rl_benchmarking.rl_benchmarker import Benchmarker


benchmarker = Benchmarker(run_name="nn_controller_adaptive")

benchmarker.train_and_evaluate(
    train_with_failures=True,
    use_pretrained=True,
    use_adaptive_approach=True,
    pretrain_only=True,
)

benchmarker.deploy_and_test(
    train_with_failures=True,
    use_pretrained=True,
    use_adaptive_approach=True,
    phase=1,
    ckpt_name="pretraining_state_adaptive.pkl",
)
