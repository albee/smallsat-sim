from experiments.rl_benchmarking.rl_benchmarker import Benchmarker


benchmarker = Benchmarker(run_name="lqr")

benchmarker.deploy_and_test_classic(controller_type="lqr")
