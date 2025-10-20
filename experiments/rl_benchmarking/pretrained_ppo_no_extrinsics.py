import os

# Set flags to improve XLA performance on GPU
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "

from experiments.rl_benchmarking.rl_benchmarker import Benchmarker


benchmarker = Benchmarker(run_name="pretrained_ppo_no_extrinsics")

benchmarker.train_and_evaluate(use_pretrained=True, use_adaptive_approach=False)
benchmarker.deploy_and_test(use_pretrained=True, use_adaptive_approach=False)
