from smallsat_sim.controllers.base_controller import BaseController
from smallsat_sim.envs.base_env import BaseEnv

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

import numpy as np
import os
import casadi as ca
import mujoco

from casadi import SX


class NominalMPCController(BaseController):
    def __init__(self, env, planner) -> None:
        # Fetch correct controller config
        self.ctrl_cfg = env.env_cfg.control.NominalMPC

        # Initialize base class
        super().__init__(env, planner, self.ctrl_cfg)

        # Generate solver
        self._generate_solver(env)

        # Initialize solver
        self._initialize_solver(env)

        # Check if there is a viewer. In case there is not,
        # dynamically allocate the visualize method to a lambda
        # function doing nothing.
        if env.viewer:
            self.viewer = env.viewer
        else:
            self._visualize = lambda *args, **kwargs: None

    def _generate_solver(self, env) -> None:
        """
        This method generates the necessary solver C code
        """
        # Initialize OCP instance
        ocp = AcadosOcp()

        # Fetch symbolic model
        model = env.symbolic_model

        # Setup acados model (interface between acados and CasADi)
        acados_model = AcadosModel()
        acados_model.f_impl_expr = model.f_impl_expr
        acados_model.f_expl_expr = model.f_expl_expr
        acados_model.x = model.x
        acados_model.xdot = model.xdot
        acados_model.u = model.u
        acados_model.z = model.z
        acados_model.p = model.p
        acados_model.name = "OCPsolver"

        # Assign parameters and model
        Ts = self.ctrl_cfg.Ts
        p = ca.vertcat(SX.sym("xref"), SX.sym("yref"), SX.sym("zref"))
        acados_model.p = p
        ocp.model = acados_model

        # Define and assign cost functions
        Q = self.ctrl_cfg.cost.Q
        R = self.ctrl_cfg.cost.R
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = (
            model.u.T @ R @ model.u + 1e-2 * model.x[10:13].T @ Q @ model.x[10:13]
        )
        ocp.model.cost_expr_ext_cost_e = (model.x[0:3] - p).T @ Q @ (model.x[0:3] - p)

        # Set OCP dimensions
        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.np = p.size()[0]
        ocp.dims.N = self.ctrl_cfg.N

        # Define state constraints
        ocp.constraints.lbx = np.array(
            [-100, -100, -100, -1.1, -1, -1, -1, -1, -1, -1, -0.5, -0.5, -0.5]
        )
        ocp.constraints.ubx = np.array(
            [100, 100, 100, 1.1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5]
        )
        ocp.constraints.idxbx = np.arange(nx)

        ocp.constraints.lbx_e = np.array(
            [-100, -100, -100, -1.1, -1, -1, -1, -1, -1, -1, -0.5, -0.5, -0.5]
        )
        ocp.constraints.ubx_e = np.array(
            [100, 100, 100, 1.1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5]
        )
        ocp.constraints.idxbx_e = np.arange(nx)

        # Define input constraints
        thruster_forces = [
            thruster.forcerange for thruster in env.model_cfg.Thrusters.thruster_list
        ]
        ocp.constraints.lbu = np.array([forces[0] for forces in thruster_forces])
        ocp.constraints.ubu = np.array([forces[1] for forces in thruster_forces])
        ocp.constraints.idxbu = np.arange(nu)

        # Set intial condition
        ocp.constraints.x0 = env.obs[0:13]
        ocp.constraints.idxbx_0 = np.arange(nx)
        ocp.parameter_values = np.zeros(ocp.dims.np)

        # Configure solver options
        ocp.solver_options.Tsim = Ts
        ocp.solver_options.tf = Ts * self.ctrl_cfg.N
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.print_level = 0

        # Set code generation directory
        save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "c_generated_code"
        )
        ocp.code_export_directory = save_dir

        # Create solver with agent specific code files
        filename = os.path.join(save_dir, "acados_pacejka_mpcc_solver_config.json")

        self.ocp_solver = AcadosOcpSolver(ocp, json_file=filename)

        print("Solver generated successfully.")

    def _initialize_solver(self, env: BaseEnv) -> np.ndarray:
        xinit = env.obs[0:13]

        [self.ocp_solver.set(i, "x", xinit) for i in range(self.ctrl_cfg.N + 1)]
        [self.ocp_solver.set(i, "u", np.zeros((12, 1))) for i in range(self.ctrl_cfg.N)]

    def get_control_input(self, env: BaseEnv) -> np.ndarray:
        """
        Calculate the control input based on current observation
        """
        # Check solver status and re-initialize if needed
        if self.ocp_solver.status != 0:
            print(f"Solution optimal, solver status: {self.ocp_solver.status}")
            self._initialize_solver(env)

        # Set the reference position
        ref_pos = self.planner.get_reference(env.obs).reshape(3, 1)
        [self.ocp_solver.set(i, "p", ref_pos) for i in range(self.ctrl_cfg.N + 1)]

        # Solve for the first control input in receding horizon fashion
        u0 = self.ocp_solver.solve_for_x0(env.obs[0:13], print_stats_on_failure=True)
        self._visualize()

        return u0

    def _visualize(self) -> None:
        """
        Plot predicted trajectory of MPC in MuJoCo viewer.
        """
        offset = self.viewer.user_scn.ngeom
        for i in range(self.ctrl_cfg.N + 1):
            point = self.ocp_solver.get(i, "x")[0:3]
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[i + offset],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.05, 0, 0],
                pos=point,
                mat=np.eye(3).flatten(),
                rgba=np.array([0, 0, 1, 2]),
            )

        self.viewer.user_scn.ngeom += self.ctrl_cfg.N + 1
