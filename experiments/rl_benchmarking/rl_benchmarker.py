import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import wandb

from smallsat_sim.utils.helpers import (
    get_args,
    calc_attitude_error,
    calc_lateral_tracking_error,
)
from smallsat_sim.utils.logger import Logger
from smallsat_sim.controllers.rl.runners.on_policy_runner import OnPolicyRunner
from smallsat_sim.envs.astrobee_rl.env import AstrobeeEnvVectorized
from smallsat_sim.envs.astrobee.env import AstrobeeEnv
from smallsat_sim.envs.disturbances import DisturbanceList, ConstantForceDisturbance
from smallsat_sim.planners.oracle.oracle_rl import OraclePlannerRL
from smallsat_sim.planners.oracle.oracle import OraclePlanner
from smallsat_sim.controllers.rl.controller import RLController
from smallsat_sim.controllers.nominal_mpc.controller import NominalMPCController
from smallsat_sim.controllers.lqr.controller import LQRController


class Benchmarker(object):
    """
    Helper functions to train and test the RL controller.
    NOTE: scripts using Benchmarker must be run in headless mode.
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
        train_with_failures: bool,
        use_pretrained: bool,
        use_adaptive_approach: bool,
        am_architecture: str | None = None,
        phase: int = 2,
        pretrain_only: bool = False,
    ) -> None:
        """
        Train and evaluate the RL controller.
        """
        # Create environment
        env = AstrobeeEnvVectorized(
            args=self.args,
            run_name=self.run_name,
            train_with_failures=train_with_failures,
            use_pretrained=use_pretrained,
            use_adaptive_approach=use_adaptive_approach,
            am_architecture=am_architecture,
        )

        # Create planner
        planner = OraclePlannerRL(env, radius=0.0)

        # Create runner
        runner = OnPolicyRunner(env, planner)

        if use_pretrained:
            # Pretraining
            runner.pretrain()

        if not pretrain_only:
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

        if env.use_wandb and wandb.run is not None:
            wandb.finish()

    def deploy_and_test(
        self,
        train_with_failures: bool,
        use_pretrained: bool,
        use_adaptive_approach: bool,
        am_architecture: str | None = None,
        phase: int = 2,
        ckpt_name: str | None = None,
        test_pd: bool = False,
    ) -> None:
        """
        Deploy and test the RL controller.
        """
        def _save_video(stage_name: str) -> None:
            if self.args.video:
                video_dir = os.path.join(
                    "experiments",
                    "rl_results",
                    self.run_name,
                    stage_name,
                    "videos",
                )
                output_name = (
                    f"{self.run_name}_{stage_name}_run{env.run_id:04d}"
                )
                env.get_sim_rendering(output_name, output_dir=video_dir)

        # Create environment
        env = AstrobeeEnvVectorized(
            args=self.args,
            run_name=self.run_name,
            init_pos=jnp.array([5.0, 0.0, 10.17]),
            max_start_offset=0.5,
            train_with_failures=train_with_failures,
            use_pretrained=use_pretrained,
            use_adaptive_approach=use_adaptive_approach,
            am_architecture=am_architecture,
        )

        # Create planner
        planner = OraclePlannerRL(env)

        # Create controller
        ctrl = RLController(env, planner, ckpt_name=ckpt_name)

        # Simulation loop
        ctrl.control(phase=phase, test_pd=test_pd)
        _save_video("deployment")

        # Test stuck off thrusters
        ctrl.control(
            stage="stuck_off_deployment",
            phase=phase,
            test_pd=test_pd,
            perturbation_distribution=jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        )
        _save_video("stuck_off_deployment")

        # Test stuck on thrusters
        ctrl.control(
            stage="stuck_on_deployment",
            phase=phase,
            test_pd=test_pd,
            perturbation_distribution=jnp.array([0.0, 1.0, 0.0, 0.0, 0.0]),
        )
        _save_video("stuck_on_deployment")

        # Test faulty valve
        ctrl.control(
            stage="faulty_valve_deployment",
            phase=phase,
            test_pd=test_pd,
            perturbation_distribution=jnp.array([0.0, 0.0, 1.0, 0.0, 0.0]),
        )
        _save_video("faulty_valve_deployment")

        # Test saturated thrust
        ctrl.control(
            stage="saturated_thrust_deployment",
            phase=phase,
            test_pd=test_pd,
            perturbation_distribution=jnp.array([0.0, 0.0, 0.0, 1.0, 0.0]),
        )
        _save_video("saturated_thrust_deployment")

        # Test thrust instability
        ctrl.control(
            stage="thrust_instability_deployment",
            phase=phase,
            test_pd=test_pd,
            perturbation_distribution=jnp.array([0.0, 0.0, 0.0, 0.0, 1.0]),
        )
        _save_video("thrust_instability_deployment")

        # Test constant force disturbances
        ctrl.control(
            stage="constant_force_disturbances_deployment",
            phase=phase,
            test_pd=test_pd,
            apply_disturbances=True,
        )
        _save_video("constant_force_disturbances_deployment")

        # Save log if logging is enabled
        if self.args.log:
            env.logger.save_log()

    def deploy_and_test_classic(self, controller_type: str) -> None:
        """
        Deploy and test classic controllers (Nominal MPC, LQR) on the oracle trajectory.
        """
        def _save_video(env, stage_name: str) -> None:
            if self.args.video:
                video_dir = os.path.join(
                    "experiments",
                    "rl_results",
                    self.run_name,
                    stage_name,
                    "videos",
                )
                output_name = f"{self.run_name}_{stage_name}_run{env.run_id:04d}"
                env.get_sim_rendering(output_name, output_dir=video_dir)

        def _log_step(env, planner, stage_name: str) -> None:
            if not hasattr(env, "logger"):
                return
            obs = env.get_obs()
            tracking_error = float(
                calc_lateral_tracking_error(obs=obs, planner=planner)
            )
            angle_error = float(
                calc_attitude_error(np.array([1, 0, 0, 0]), obs[3:7])
            )
            env.logger.log(
                env.run_id,
                float(env.data.time),
                run_name=self.run_name,
                stage=stage_name,
                mean_tracking_error=tracking_error,
                mean_angle_error=angle_error,
                mean_extrinsic_error=0.0,
            )

        def _apply_stage_perturbation(env, stage_name: str, start_time: float) -> None:
            if stage_name == "stuck_off_deployment":
                env.perturbations.perturbations[0].stuck_off_thruster(
                    index=0, start_time=start_time
                )
            elif stage_name == "stuck_on_deployment":
                env.perturbations.perturbations[1].stuck_on_thruster(
                    index=0, start_time=start_time
                )
            elif stage_name == "faulty_valve_deployment":
                env.perturbations.perturbations[2].register_perturbation(
                    index=0, start_time=start_time
                )
            elif stage_name == "saturated_thrust_deployment":
                env.perturbations.perturbations[3].register_perturbation(
                    index=0, start_time=start_time
                )
            elif stage_name == "thrust_instability_deployment":
                env.perturbations.perturbations[4].register_perturbation(
                    index=0, start_time=start_time
                )
            elif stage_name == "constant_force_disturbances_deployment":
                disturbance_key = jax.random.PRNGKey(0)
                env.disturbances = DisturbanceList(
                    [ConstantForceDisturbance(env.env_cfg, disturbance_key)]
                )
                env.disturbances.disturbances[0].const_force_disturbance(
                    start_time=start_time
                )

        if controller_type not in {"nominal_mpc", "lqr"}:
            raise ValueError(
                "controller_type must be one of: 'nominal_mpc', 'lqr'."
            )

        # Create environment
        env = AstrobeeEnv(args=self.args)

        # Create planner (oracle trajectory matching RL layout)
        planner = OraclePlanner(
            env,
            radius=3.0,
            spacing=1.0,
            clearance_dist=0.2,
            plane="xy",
            z_offset=10.17,
        )

        # Create controller
        if controller_type == "nominal_mpc":
            ctrl = NominalMPCController(env, planner)
        else:
            ctrl = LQRController(env, planner)

        stages = [
            "deployment",
            "stuck_off_deployment",
            "stuck_on_deployment",
            "faulty_valve_deployment",
            "saturated_thrust_deployment",
            "thrust_instability_deployment",
            "constant_force_disturbances_deployment",
        ]

        for stage_name in stages:
            env.reset()
            env.reset_perturbations()
            env.disturbances = None
            planner.idx_reference_point = 0

            env.reset_to_state(
                pos=np.array([5.0, 0.0, 10.17]),
                att=np.array([0.0, 0.0, 0.0]),
            )

            step = 0
            perturbation_applied = False
            while env.data.time <= env.env_cfg.sim.max_sim_time:
                if step == 100 and not perturbation_applied:
                    _apply_stage_perturbation(env, stage_name, float(env.data.time))
                    perturbation_applied = True

                ctrl_input = ctrl.get_control_input(env)
                env.step(input=ctrl_input)
                _log_step(env, planner, stage_name)
                step += 1

            _save_video(env, stage_name)

        # Save log if logging is enabled
        if self.args.log:
            env.logger.save_log()

        env.close()
