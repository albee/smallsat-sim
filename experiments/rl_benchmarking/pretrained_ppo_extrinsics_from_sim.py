import os

from experiments.rl_benchmarking.rl_benchmarker import Benchmarker

# Set flags to improve XLA performance on GPU
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true " "--xla_gpu_triton_gemm_any=True "
)

benchmarker = Benchmarker(run_name="pretrained_ppo_extrinsics_from_sim")

benchmarker.train_and_evaluate(use_pretrained=True, use_adaptive_approach=True, phase=1)
benchmarker.deploy_and_test(use_pretrained=True, use_adaptive_approach=True, phase=1)
