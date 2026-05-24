import os
from pathlib import Path

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
from smallsat_sim.envs.astrobee_benchmark.env import AstrobeeBenchmarkEnv
from smallsat_sim.envs.disturbances import DisturbanceList, ConstantForceDisturbance
from smallsat_sim.planners.oracle.oracle_rl import OraclePlannerRL
from smallsat_sim.planners.oracle.oracle import OraclePlanner
from smallsat_sim.controllers.rl.controller import RLController
from smallsat_sim.controllers.lqr.controller import LQRController
from smallsat_sim.envs.astrobee_rl.cfg import config as rl_config


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
        self._repo_root = Path(__file__).resolve().parents[2]

    def train_and_evaluate(
        self,
        train_with_failures: bool,
        use_pretrained: bool,
        use_adaptive_approach: bool,
        am_architecture: str | None = None,
        adaptive_context_mode: str | None = None,
        use_task_conditioned_am: bool | None = None,
        am_predict_delta_weight: float | None = None,
        am_predict_tracking_weight: float | None = None,
        am_predict_authority_weight: float | None = None,
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
            adaptive_context_mode=adaptive_context_mode,
            use_task_conditioned_am=use_task_conditioned_am,
            am_predict_delta_weight=am_predict_delta_weight,
            am_predict_tracking_weight=am_predict_tracking_weight,
            am_predict_authority_weight=am_predict_authority_weight,
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

            if os.environ.get("SMALLSAT_SKIP_EVAL", "0") == "1":
                print(
                    "Skipping policy evaluation because SMALLSAT_SKIP_EVAL=1. "
                    "Set RUN_EVAL=1 in train_test_all.sh to enable it."
                )
            else:
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
        adaptive_context_mode: str | None = None,
        use_task_conditioned_am: bool | None = None,
        am_predict_delta_weight: float | None = None,
        am_predict_tracking_weight: float | None = None,
        am_predict_authority_weight: float | None = None,
        phase: int = 2,
        ckpt_name: str | None = None,
        test_pd: bool = False,
    ) -> None:
        """
        Deploy and test the RL controller.
        """
        if os.environ.get("SMALLSAT_SKIP_DEPLOY", "0") == "1":
            print(
                "Skipping deployment because SMALLSAT_SKIP_DEPLOY=1. "
                "Set RUN_DEPLOY=1 in train_test_all.sh to run deployment tests."
            )
            return

        rl_cfg = rl_config.EnvConfig().control.RL

        def _save_video(stage_name: str) -> None:
            if self.args.video:
                video_dir = os.path.join(
                    str(self._repo_root),
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
            init_pos=jnp.array(rl_cfg.deployment_init_pos, dtype=jnp.float32),
            max_start_offset=rl_cfg.deployment_max_start_offset,
            train_with_failures=train_with_failures,
            use_pretrained=use_pretrained,
            use_adaptive_approach=use_adaptive_approach,
            am_architecture=am_architecture,
            adaptive_context_mode=adaptive_context_mode,
            use_task_conditioned_am=use_task_conditioned_am,
            am_predict_delta_weight=am_predict_delta_weight,
            am_predict_tracking_weight=am_predict_tracking_weight,
            am_predict_authority_weight=am_predict_authority_weight,
        )

        # Create planner
        planner = OraclePlannerRL(
            env,
            radius=rl_cfg.deployment_radius,
            spacing=rl_cfg.deployment_spacing,
        )

        # Create controller
        ctrl = RLController(env, planner, ckpt_name=ckpt_name)

        def _run_deployment_stage(stage_name: str = "deployment", **control_kwargs):
            print(f"[Deployment] Running {self.run_name}: {stage_name}", flush=True)
            ctrl.control(stage=stage_name, phase=phase, test_pd=test_pd, **control_kwargs)
            _save_video(stage_name)

        # Simulation loop
        _run_deployment_stage("deployment")

        # Test stuck off thrusters
        _run_deployment_stage(
            "stuck_off_deployment",
            perturbation_distribution=jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        )

        # Test stuck on thrusters
        _run_deployment_stage(
            "stuck_on_deployment",
            perturbation_distribution=jnp.array([0.0, 1.0, 0.0, 0.0, 0.0]),
        )

        # Test faulty valve
        _run_deployment_stage(
            "faulty_valve_deployment",
            perturbation_distribution=jnp.array([0.0, 0.0, 1.0, 0.0, 0.0]),
        )

        # Test saturated thrust
        _run_deployment_stage(
            "saturated_thrust_deployment",
            perturbation_distribution=jnp.array([0.0, 0.0, 0.0, 1.0, 0.0]),
        )

        # Test thrust instability
        _run_deployment_stage(
            "thrust_instability_deployment",
            perturbation_distribution=jnp.array([0.0, 0.0, 0.0, 0.0, 1.0]),
        )

        # Test constant force disturbances
        _run_deployment_stage(
            "constant_force_disturbances_deployment",
            apply_disturbances=True,
        )

        stuck_off_dist = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
        stuck_on_dist = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0])

        # Stress tests: compound/severity-shift failures not used as core training changes.
        _run_deployment_stage(
            "two_stuck_off_deployment",
            perturbation_distributions=(stuck_off_dist, stuck_off_dist),
        )

        _run_deployment_stage(
            "two_stuck_on_deployment",
            perturbation_distributions=(stuck_on_dist, stuck_on_dist),
        )

        _run_deployment_stage(
            "stuck_off_plus_disturbance_deployment",
            perturbation_distribution=stuck_off_dist,
            apply_disturbances=True,
        )

        _run_deployment_stage(
            "stuck_on_plus_disturbance_deployment",
            perturbation_distribution=stuck_on_dist,
            apply_disturbances=True,
        )

        _run_deployment_stage(
            "mixed_stuck_off_stuck_on_deployment",
            perturbation_distributions=(stuck_off_dist, stuck_on_dist),
        )

        # Single-life sequence: multiple failures/disturbances in one deployment
        # without resetting the spacecraft.
        _run_deployment_stage(
            "single_life_fault_sequence_deployment",
            perturbation_sequence=(
                (100, stuck_off_dist),
                (200, stuck_on_dist),
            ),
            disturbance_start_steps=(300,),
        )

        # Save log if logging is enabled
        if self.args.log:
            env.logger.save_log()

    def deploy_and_test_classic(self, controller_type: str) -> None:
        """
        Deploy and test classic controllers (Nominal MPC, LQR) on the oracle trajectory.
        """
        rl_cfg = rl_config.EnvConfig().control.RL

        def _resolve_deployment_len() -> int | None:
            if hasattr(env.env_cfg.control, "RL"):
                return env.env_cfg.control.RL.deployment_len
            try:
                from smallsat_sim.envs.astrobee_rl.cfg import config as rl_config
            except ImportError:
                return None
            return rl_config.EnvConfig().control.RL.deployment_len

        def _seed_for_stage(stage_name: str) -> jax.random.PRNGKey:
            base_seed = int(getattr(env.env_cfg.sim, "seed", 0))
            stage_idx = stages.index(stage_name)
            seed = base_seed + stage_idx
            np.random.seed(seed)
            return jax.random.PRNGKey(seed)

        def _save_video(env, stage_name: str) -> None:
            if self.args.video:
                video_dir = os.path.join(
                    str(self._repo_root),
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
                mean_lateral_error=tracking_error,
                mean_angle_error=angle_error,
                mean_extrinsic_error=0.0,
            )

        def _apply_stage_perturbation(env, stage_name: str, start_time: float) -> None:
            if stage_name == "stuck_off_deployment":
                env.perturbations.perturbations[0].stuck_off_thruster(
                    index=None, start_time=start_time
                )
            elif stage_name == "stuck_on_deployment":
                env.perturbations.perturbations[1].stuck_on_thruster(
                    index=None, start_time=start_time
                )
            elif stage_name == "faulty_valve_deployment":
                env.perturbations.perturbations[2].register_perturbation(
                    index=None, start_time=start_time
                )
            elif stage_name == "saturated_thrust_deployment":
                env.perturbations.perturbations[3].register_perturbation(
                    index=None, start_time=start_time
                )
            elif stage_name == "thrust_instability_deployment":
                env.perturbations.perturbations[4].register_perturbation(
                    index=None, start_time=start_time
                )
            elif stage_name == "constant_force_disturbances_deployment":
                disturbance_key = _seed_for_stage(stage_name)
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
        env = AstrobeeBenchmarkEnv(args=self.args)

        # Create planner (oracle trajectory matching RL layout)
        planner = OraclePlanner(
            env,
            radius=rl_cfg.deployment_radius,
            spacing=rl_cfg.deployment_spacing,
            clearance_dist=0.2,
            plane="xy",
            z_offset=10.17,
        )

        # Create controller
        if controller_type == "nominal_mpc":
            from smallsat_sim.controllers.nominal_mpc.controller import (
                NominalMPCController,
            )

            ctrl = NominalMPCController(env, planner)
        else:
            # Benchmark override: use MPC-style costs so LQR actually tracks in nominal runs.
            env.env_cfg.control.LQR.cost.Q = (
                env.env_cfg.control.NominalMPC.cost.Q.copy()
            )
            env.env_cfg.control.LQR.cost.R = (
                env.env_cfg.control.NominalMPC.cost.R.copy()
            )
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
            _seed_for_stage(stage_name)

            env.reset_to_state(
                pos=np.array(rl_cfg.deployment_init_pos),
                att=np.array([0.0, 0.0, 0.0]),
            )

            step = 0
            perturbation_applied = False
            max_steps = _resolve_deployment_len()
            while env.data.time <= env.env_cfg.sim.max_sim_time:
                if max_steps is not None and step >= int(max_steps):
                    break
                if step == 100 and not perturbation_applied:
                    _apply_stage_perturbation(env, stage_name, float(env.data.time))
                    perturbation_applied = True

                ctrl_input = ctrl.get_control_input(env)
                env.step(input=ctrl_input)
                if self.args.video:
                    if (
                        env.data.time >= env.env_cfg.renderer.start_recording
                        and env.data.time <= env.env_cfg.renderer.end_recording
                    ):
                        env._update_renderer()
                _log_step(env, planner, stage_name)
                step += 1

            _save_video(env, stage_name)

        # Save log if logging is enabled
        if self.args.log:
            env.logger.save_log()

        env.close()
