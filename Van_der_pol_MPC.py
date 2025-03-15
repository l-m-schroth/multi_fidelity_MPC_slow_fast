from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class VanDerPolMPCOptions:
    # System Parameters
    mu: float = 1.0
    epsilon_1: float = 0.005
    epsilon_2: float = 0.3
    epsilon_3: float = 0.001
    epsilon_4: float = 0.01
    d1: float = 0.2
    d2: float = 0.1
    d3: float = 0.3

    # MPC Parameters
    step_size_list: List[float] = [0.02] * 10  # Define default in __post_init__
    N: int = 10
    switch_stage: int = 5
    nlp_solver_type: str = "SQP"  # "SQP_RTI" is another option
    qp_solver: str = "FULL_CONDENSING_HPIPM"

    # Cost Weights (Only for x1, x2 tracking)
    Q_2d: np.ndarray = np.diag([1.0, 1.0])  # Track x1 and x2
    R_2d: np.ndarray = np.diag([1.0, 1.0])   # Control cost


from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver, AcadosMultiphaseOcp
import casadi as ca

class VanDerPolMPC:

    def __init__(self, options: VanDerPolMPCOptions):
        """
        Initializes the MPC problem using the provided options.
        """
        self.opts = options  # Store options as self.opts

        # Create a 2-phase OCP (3D → 2D transition)
        self._create_multiphase_ocp()

        # Create Acados Solver
        json_file = f"acados_ocp_vdp_{self.opts.switch_stage}.json"
        self.ocp_solver = AcadosOcpSolver(self.mpc_ocp, json_file=json_file)

        # Create Acados Simulation Solver (for closed-loop simulation)
        self.sim_solver_3d = self._create_sim_3d(json_file_suffix="vdp_3d_sim")

    def _create_multiphase_ocp(self):
        """
        Creates a 2-phase AcadosMultiphaseOcp:
          - Phase 0: 3D Model (x1, x2, x3) until switch_stage
          - Phase 1: 2D Model (x1, x2) afterward
        """
        switch_stage = self.opts.switch_stage
        N = self.opts.switch_stage

        # Multi-phase OCP setup
        self.mpc_ocp = AcadosMultiphaseOcp(N_list=[switch_stage, N - switch_stage])
        self.mpc_ocp.solver_options.tf = sum(self.opts.step_size_list)

        # Create phases
        model_3d = self._create_vdp_3d_model
        model_2d = self._create_vdp_2d_model
        ocp_phase_0 = self._build_ocp(switch_stage, model_3d, self.opts.step_size_list[:self.opts.switch_stage])
        ocp_phase_1 = self._build_ocp(N - switch_stage, model_2d, self.opts.step_size_list[self.opts.switch_stage:])

        # Assign phases to OCP
        self.mpc_ocp.set_phase(ocp_phase_0, 0)
        self.mpc_ocp.set_phase(ocp_phase_1, 1)

        # Assign time steps
        self.mpc_ocp.solver_options.time_steps = self.opts.step_size_list

        # Set solver options
        self.mpc_ocp.solver_options.nlp_solver_type = self.opts.nlp_solver_type
        self.mpc_ocp.solver_options.qp_solver = self.opts.qp_solver
        self.mpc_ocp.solver_options.integrator_type = ["RK4", "RK4"]

    def _build_ocp_3d(self, N_phase):
        """
        Defines the first-phase OCP using the 3D model.
        """
        ocp_3d = AcadosOcp()
        ocp_3d.model = self._create_vdp_3d_model()

        nx_3d = 3
        nu_3d = 1
        ny_3d = 2 + nu_3d  # Only tracking x1, x2 + control u

        ocp_3d.cost.cost_type = "NONLINEAR_LS"
        ocp_3d.cost.cost_type_e = "NONLINEAR_LS"

        x1, x2 = ocp_3d.model.x[0], ocp_3d.model.x[1]
        u_3d = ocp_3d.model.u[0]

        ocp_3d.model.cost_y_expr = ca.vertcat(x1, x2, u_3d)
        ocp_3d.model.cost_y_expr_e = ca.vertcat(x1, x2)

        W_3d = np.block([
            [self.opts.Q_2d,              np.zeros((2,1))],
            [np.zeros((1,2)),             self.opts.R_2d]
        ])

        ocp_3d.cost.W = W_3d
        ocp_3d.cost.W_e = self.opts.Q_2d

        # initialize desired trajectory with zeros
        ocp_3d.cost.yref = np.zeros(ny_3d)
        ocp_3d.cost.yref_e = np.zeros(2)

        ocp_3d.dims.N = N_phase
        ocp_3d.solver_options.tf = sum(self.opts.step_size_list[:N_phase])
        ocp_3d.solver_options.integrator_type = "RK4"

        return ocp_3d
    
    def _build_ocp(self, model, N_phase, step_size_list):
        """
        Defines the second-phase OCP using the 2D model (neglecting x3).
        """
        ocp = AcadosOcp()
        ocp.model = model

        ny_x = 2
        nu = 2
        ny = ny_x + nu

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        x1, x2 = ocp.model.x[0], ocp.model.x[1]
        u1, u2 = ocp.model.u[0], ocp.model.u[1]

        ocp.model.cost_y_expr = ca.vertcat(x1, x2, u1, u2)
        ocp.model.cost_y_expr_e = ca.vertcat(x1, x2)

        W_2d = np.block([
            [self.opts.Q_2d,              np.zeros((2,2))],
            [np.zeros((2,2)),             self.opts.R_2d]
        ])

        ocp.cost.W = W_2d
        ocp.cost.W_e = self.opts.Q_2d

        # reference values are overwritten later
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_x)

        # ocp.dims.N = N_phase
        # ocp.solver_options.tf = sum(self.opts.step_size_list[self.opts.switch_stage:])
        # ocp.solver_options.integrator_type = "RK4"
    
    def _create_vdp_3d_model(self):
        """
        Defines the 3D Van der Pol model in CasADi/Acados form:
        - State: x = [x1, x2, x3]
        - Control: u
        - Dynamics:
            x1_dot = x2 + d1*u + epsilon_1*x3
            x2_dot = mu*(1 - x1^2)*x2 - x1 + epsilon_2*x2 + epsilon_3*x3 + d2*u
            x3_dot = - (1/epsilon_4) * x3 + d3*u
        """
        x1 = ca.SX.sym("x1")
        x2 = ca.SX.sym("x2")
        x3 = ca.SX.sym("x3")
        x = ca.vertcat(x1, x2, x3)

        u1 = ca.SX.sym("u1")  
        u2 = ca.SX.sym("u2")  
        u = ca.vertcat(u1, u2)

        # Define dynamics
        dx1 = x2 + self.opts.d1 * u1 + self.opts.epsilon_1 * x3
        dx2 = self.opts.mu * (1 - x1**2) * x2 - x1 + self.opts.epsilon_2 * x2 + self.opts.epsilon_3 * x3 + self.opts.d2 * u2
        dx3 = - (1.0 / self.opts.epsilon_4) * x3 + self.opts.d3 * u2

        xdot = ca.vertcat(dx1, dx2, dx3)

        model = AcadosModel()
        model.name = "vdp_3d"
        model.x = x
        model.u = u
        model.f_expl_expr = xdot
        model.f_impl_expr = xdot - ca.vertcat(x1, x2, x3)  # Implicit form

        return model

    def _create_vdp_2d_model(self):
        """
        Defines the 2D Van der Pol model (neglecting x3) in CasADi/Acados form:
        - State: x = [x1, x2]
        - Control: u
        - Dynamics:
            x1_dot = x2 + d1*u
            x2_dot = mu*(1 - x1^2)*x2 - x1 + epsilon_2*x2 + d2*u
        """
        x1 = ca.SX.sym("x1")
        x2 = ca.SX.sym("x2")
        x = ca.vertcat(x1, x2)

        u1 = ca.SX.sym("u1")  
        u2 = ca.SX.sym("u2")  
        u = ca.vertcat(u1, u2)

        dx1 = x2 + self.opts.d1 * u1
        dx2 = self.opts.mu * (1 - x1**2) * x2 - x1 + self.opts.epsilon_2 * x2 + self.opts.d2 * u2

        xdot = ca.vertcat(dx1, dx2)

        model = AcadosModel()
        model.name = "vdp_2d"
        model.x = x
        model.u = u
        model.f_expl_expr = xdot
        model.f_impl_expr = xdot - ca.vertcat(x1, x2)

        return model

    def _create_sim_3d(self, json_file_suffix="vdp_3d_sim"):
        """
        Creates an AcadosSim solver for the 3D model (useful for closed-loop simulation).
        """
        model_3d = self._create_vdp_3d_model()
        sim = AcadosSim()
        sim.model = model_3d

        # Pick a step size for simulation (same as the first step size)
        sim.solver_options.T = self.opts.step_size_list[0]
        sim.solver_options.integrator_type = "RK4"
        
        sim_solver = AcadosSimSolver(sim, json_file=f"acados_sim_solver_{json_file_suffix}.json")
        return sim_solver

    def set_initial_state(self, x0):
        self.acados_ocp_solver.set(0, "lbx", x0)
        self.acados_ocp_solver.set(0, "ubx", x0)

    def set_reference_trajectory(self, t0):
        """
        Set reference trajectory in the solver based on self.Phi_t.
        The function self.Phi_t(t) should return the desired reference [x1_ref, x2_ref] at time t.
        """
        N = len(self.opts.step_size_list)  # Horizon length
        t_eval = t0  # Start from the given time

        for stage in range(N):
            t_eval += self.opts.step_size_list[stage]
            x1_ref, x2_ref = self.Phi_t(t_eval)  # Get reference at time t_eval

            # Construct yref: [x1_ref, x2_ref, u_ref=0]
            y_ref = np.array([x1_ref, x2_ref, 0.0])
            self.ocp_solver.set(stage, "yref", y_ref)

        # Set terminal reference
        t_eval += self.opts.step_size_list[-1]
        x1_ref, x2_ref = self.Phi_t(t_eval)
        y_ref_N = np.array([x1_ref, x2_ref])
        self.ocp_solver.set(N, "yref", y_ref_N)

    def solve(self, x0, t):
        """
        Solves the MPC problem:
        1. Sets the initial guess with x0.
        2. Sets the reference trajectory using self.Phi_t at time t.
        3. Solves the OCP.
        """
        # Set initial guess
        self.set_initial_state(x0)

        # Set reference trajectory
        self.set_reference_trajectory(t)

        # Solve the OCP
        status = self.ocp_solver.solve()
        
        if status != 0:
            print(f"[VanDerPolMPC] OCP solver returned status {status}.")

        # Return first control input
        return self.ocp_solver.get(0, "u")
