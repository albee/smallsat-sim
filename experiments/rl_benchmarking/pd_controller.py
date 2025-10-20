import os

# Set flags to improve XLA performance on GPU
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "

from experiments.rl_benchmarking.rl_benchmarker import Benchmarker


benchmarker = Benchmarker()

benchmarker.deploy_and_test(
    use_pretrained=False, use_adaptive_approach=False, test_pd=True
)
