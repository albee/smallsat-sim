import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb

from smallsat_sim.utils.helpers import get_args
from smallsat_sim.utils.logger import Logger
from smallsat_sim.controllers.rl.runners.on_policy_runner import OnPolicyRunner
from smallsat_sim.envs.astrobee_rl.env import AstrobeeEnvVectorized
from smallsat_sim.planners.oracle.oracle_rl import OraclePlannerRL
from smallsat_sim.controllers.rl.controller import RLController


class Benchmarker(object):
    """
    Helper functions to train and test the RL controller.
    """

    def __init__(self, run_name: str = "default") -> None:
        """
        Initialize the Benchmarking class.
        """
        # Get arguments for script execution
        self.args = get_args()

        # Run name for logging
        self.run_name = run_name

    def train_and_evaluate(
        self,
        use_pretrained: bool,
        use_adaptive_approach: bool,
        phase: int = 2,
    ) -> None:
        """
        Train and evaluate the RL controller.
        """
        # Create environment
        env = AstrobeeEnvVectorized(
            args=self.args,
            run_name=self.run_name,
            use_pretrained=use_pretrained,
            use_adaptive_approach=use_adaptive_approach,
        )

        # Create planner
        planner = OraclePlannerRL(env, radius=0.0)

        # Create runner
        runner = OnPolicyRunner(env, planner)

        # Pretraining
        runner.pretrain()

        # Learning
        runner.learn()

        # Adaptation module training
        if use_adaptive_approach and phase == 2:
            runner.train_adaptation_module_on_policy()

        # Evaluation
        runner.evaluate(phase=phase)

        # Save log if logging is enabled
        if self.args.log:
            env.logger.save_log()

        # Stop wandb run
        wandb.finish()

    def deploy_and_test(
        self,
        use_pretrained: bool,
        use_adaptive_approach: bool,
        phase: int = 2,
        ckpt_name: str | None = None,
        test_pd: bool = False,
    ) -> None:
        """
        Deploy and test the RL controller.
        """
        # Create environment
        env = AstrobeeEnvVectorized(
            args=self.args,
            run_name=self.run_name,
            init_pos=jnp.array([5.0, 0.0, 10.17]),
            max_start_offset=0.5,
            use_pretrained=use_pretrained,
            use_adaptive_approach=use_adaptive_approach,
        )

        # Create planner
        planner = OraclePlannerRL(env)

        # Create controller
        ctrl = RLController(env, planner, ckpt_name=ckpt_name)

        # Simulation loop
        ctrl.control(phase=phase, test_pd=test_pd)

        # Save log if logging is enabled
        if self.args.log:
            env.logger.save_log()
