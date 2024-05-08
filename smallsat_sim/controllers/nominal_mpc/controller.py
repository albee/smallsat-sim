from smallsat_sim.controllers.base_controller import BaseController
from smallsat_sim.envs.base_env import BaseEnv

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

import numpy as np
import os
import casadi as ca
import mujoco

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

        # Check if there is a viewer. In case there is not,
        # dynamically allocate the visualize method to a lambda
        # function doing nothing.
        if env.viewer:
            self.viewer = env.viewer
        else:
            self._visualize = lambda *args, **kwargs: None

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

        Ts = 0.05

        # Define parameters
        xref = SX.sym("xref")
        yref = SX.sym("yref")
        zref = SX.sym("zref")

        p = ca.vertcat(xref, yref, zref)

        acados_model.p = p

        ocp.model = acados_model

        # Define cost
        Q = SX.eye(3)
        R = 1e-3 * SX.eye(12)

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = model.u.T @ R @ model.u + 1e-2 * model.x[10:13].T @ Q @ (model.x[10:13])
        ocp.model.cost_expr_ext_cost_e = (model.x[0:3]-p).T @ Q @ (model.x[0:3] - p)

        # ocp.model.cost_expr_ext_cost = 0
        # ocp.model.cost_expr_ext_cost_e = 100*(model.x[0:3]-p).T @ Q @ (model.x[0:3] - p)

        # Setup OCP
        ocp.dims.nx = nx
        ocp.dims.np = 3
        ocp.dims.ny = ny
        ocp.dims.ny_e = ny_e
        ocp.dims.nbx = nx
        ocp.dims.nbx_e = nx
        ocp.dims.nbu = nu
        ocp.dims.nu = nu
        ocp.dims.N = self.ctrl_cfg.N
        ocp.dims.nh = 0
        ocp.dims.nsh = 0


        # Define state constraints
        ocp.constraints.lbx = np.array([-100, -100, -100, -1.1, -1, -1, -1, -1, -1, -1, -0.5, -0.5, -0.5])
        ocp.constraints.ubx = np.array([100, 100, 100, 1.1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5])
        ocp.constraints.idxbx = np.arange(nx)

        ocp.constraints.lbx_e = np.array([-100, -100, -100, -1.1, -1, -1, -1, -1, -1, -1, -0.5, -0.5, -0.5])
        ocp.constraints.ubx_e = np.array([100, 100, 100, 1.1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5])
        ocp.constraints.idxbx_e = np.arange(nx)


        # Define input constraints
        lbu = []
        ubu = []
        for thruster in (env.model_cfg.Thrusters.thruster_list):
            lbu.append(thruster.forcerange[0])
            ubu.append(thruster.forcerange[1])

        ocp.constraints.lbu = np.array(lbu)
        ocp.constraints.ubu = np.array(ubu)
        ocp.constraints.idxbu = np.arange(nu)

        # set intial condition
        ocp.constraints.x0 = env.obs[0:13]
        ocp.constraints.idxbx_0 = np.arange(nx)
        ocp.parameter_values = np.zeros(ocp.dims.np)
        # set QP solver and integration
        ocp.solver_options.Tsim = Ts
        ocp.solver_options.tf = Ts * self.ctrl_cfg.N
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        #ocp.solver_options.sim_method_num_stages = 1
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
        [self.ocp_solver.set(i, "u", np.zeros((12,1))) for i in range(self.ctrl_cfg.N)]


    def get_control_input(self, env: BaseEnv) -> np.ndarray:
        """
        Calculate the control input based on current obs
        """
        if self.ocp_solver.status != 0:
            self._initialize_solver(env)

        ref_pos = self.planner.get_reference(env.obs).reshape(3, 1)
        [self.ocp_solver.set(i, "p", ref_pos) for i in range(self.ctrl_cfg.N + 1)]
        u0 = self.ocp_solver.solve_for_x0(env.obs[0:13], print_stats_on_failure=True)
        print(f"Solved with status {self.ocp_solver.status}")
        self._visualize()

        print(f"Quaternion: {env.obs[3:7]}")
        print(f"Length Quaternion: {env.obs[3]**2 + env.obs[4]**2 + env.obs[5]**2 + env.obs[6]**2}")
        print(f"omega x: {env.obs[10]}")
        print(f"omega y: {env.obs[11]}")
        print(f"omega z: {env.obs[12]}")

        print(f"Input: {u0}")

        print(f"Reference: {ref_pos}")
        print(f"Current pos: {self.ocp_solver.get(0, 'x')}")
        print(f"Current pos @N: {self.ocp_solver.get(self.ctrl_cfg.N, 'x')}")
        

        return u0

    def _visualize(self):
        """
        Plot predicted trajectory of MPC in MuJoCo viewer.
        """
        prediction = []
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

        print(f"omega predicted: {self.ocp_solver.get(1, 'x')[10:13]}")
        print(f"omega predicted N: {self.ocp_solver.get(self.ctrl_cfg.N, 'x')[10:13]}")
