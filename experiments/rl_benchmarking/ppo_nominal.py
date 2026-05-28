import os

# Set flags to improve XLA performance on GPU
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "

from experiments.rl_benchmarking.rl_benchmarker import Benchmarker
from smallsat_sim.envs.astrobee_rl.cfg import config as rl_config


rl_config.EnvConfig.control.RL.PPO.epochs = 600


benchmarker = Benchmarker(run_name="ppo_nominal")

benchmarker.train_and_evaluate(
    train_with_failures=False,
    use_pretrained=False,
    use_adaptive_approach=False,
    adaptive_policy_mode="direct",
    context_fusion="film",
    nominal_ckpt_export_name=os.environ.get(
        "SMALLSAT_NOMINAL_CKPT_ALIAS", "nominal_actor_baseline.pkl"
    ),
)

benchmarker.deploy_and_test(
    train_with_failures=False,
    use_pretrained=False,
    use_adaptive_approach=False,
    adaptive_policy_mode="direct",
    context_fusion="film",
)
