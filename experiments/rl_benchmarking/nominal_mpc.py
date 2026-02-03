from experiments.rl_benchmarking.rl_benchmarker import Benchmarker


benchmarker = Benchmarker(run_name="nominal_mpc")

benchmarker.deploy_and_test_classic(controller_type="nominal_mpc")
