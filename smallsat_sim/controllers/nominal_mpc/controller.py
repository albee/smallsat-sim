from smallsat_sim.controllers.base_controller import BaseController
from smallsat_sim.envs.base_env import BaseEnv

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

import numpy as np
import os
import casadi as ca

from casadi import SX, DM


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

    def _generate_solver(self, env) -> None:
        # Create solver interface
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

        # Set dimensions
        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = 0
        ny_e = 0

        # Define constraints

        # Define parameters
        xref = SX.sym("xref")
        yref = SX.sym("yref")
        zref = SX.sym("zref")

        p = ca.vertcat(xref, yref, zref)

        acados_model.p = p

        ocp.model = acados_model

        # Define cost
        Q = 1e-3 * np.eye(3)
        R = 1e-3 * np.eye(12)

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = model.u.T @ R @ model.u
        # ocp.model.cost_expr_ext_cost_e = model.x[0:3].T @ Q @ model.x[0:3]
        ocp.model.cost_expr_ext_cost_e = (
            (model.x[0] - xref) ** 2
            + (model.x[1] - yref) ** 2
            + (model.x[2] - zref) ** 2
        )

        # Setup OCP
        ocp.dims.nx = nx
        ocp.dims.np = 3
        ocp.dims.ny = ny
        ocp.dims.ny_e = ny_e
        ocp.dims.nbx = 0
        ocp.dims.nbu = 0
        ocp.dims.nu = nu
        ocp.dims.N = self.ctrl_cfg.N
        ocp.dims.nh = 0
        ocp.dims.nsh = 0

        # set intial condition

        ocp.constraints.x0 = env.obs[0:13]
        ocp.constraints.idxbx_0 = np.arange(nx)
        ocp.parameter_values = np.zeros(ocp.dims.np)
        # set QP solver and integration
        ocp.solver_options.Tsim = 0.04
        ocp.solver_options.tf = 0.04 * 40
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 3
        ocp.solver_options.print_level = 0

        # Set save paths
        save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "c_generated_code"
        )
        ocp.code_export_directory = save_dir

        # create solver with agent specific code files
        filename = os.path.join(save_dir, "acados_pacejka_mpcc_solver_config.json")

        self.ocp_solver = AcadosOcpSolver(ocp, json_file=filename)

        print("Solver generated successfully.")

    def _initialize_solver(self, env: BaseEnv) -> np.ndarray:
        xinit = env.obs[0:13]

        [self.ocp_solver.set(i, "x", xinit) for i in range(self.ctrl_cfg.N + 1)]

    def get_control_input(self, env: BaseEnv) -> np.ndarray:
        """
        Calculate the control input based on current obs
        """
        ref_pos = self.planner.get_reference(env.obs).reshape(3, 1)
        self.ocp_solver.set(self.ctrl_cfg.N, "p", ref_pos)
        u0 = self.ocp_solver.solve_for_x0(env.obs[0:13])
        print(f"Solved with status {self.ocp_solver.status}")

        return u0
