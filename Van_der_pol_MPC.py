from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass
class VanDerPolMPCOptions:
    # System Parameters
    mu: float = 1.0
    epsilon_1: float = 10
    epsilon_2: float = 0.3
    epsilon_3: float = 5
    epsilon_4: float = 0.01
    d1: float = 1.6
    d2: float = 1.6
    d3: float = 5

    # MPC Parameters
    step_size_list: List[float] = field(default_factory=lambda: [0.02] * 40)  # Corrected
    N: int = 40
    switch_stage: int = 20
    nlp_solver_type: str = "SQP"  # "SQP_RTI" is another option
    qp_solver: str = "FULL_CONDENSING_HPIPM"

    # Cost Weights (Only for x1, x2 tracking)
    Q_2d: np.ndarray = field(default_factory=lambda: np.diag([10.0, 10.0]))  # Corrected
    R_2d: np.ndarray = field(default_factory=lambda: np.diag([0.001, 0.001]))  # Corrected



from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver, AcadosMultiphaseOcp
import casadi as ca

class VanDerPolMPC:

    def __init__(self, options: VanDerPolMPCOptions, Phi_t):
        """
        Initializes the MPC problem using the provided options.
        """
        self.opts = options  # Store options as self.opts
        self.Phi_t = Phi_t

        # Create a 2-phase OCP (3D → 2D transition)
        self._create_multiphase_ocp()

        # Create Acados Solver
        json_file = f"acados_ocp_vdp_{self.opts.switch_stage}.json"
        self.acados_ocp_solver = AcadosOcpSolver(self.mp_ocp, json_file=json_file)

        # Create Acados Simulation Solver (for closed-loop simulation)
        self.acados_sim_solver_3d = self._create_sim_3d(json_file_suffix="vdp_3d_sim")

    def _create_multiphase_ocp(self):
        """
        Creates a 2-phase AcadosMultiphaseOcp:
          - Phase 0: 3D Model (x1, x2, x3) until switch_stage
          - Phase 1: 2D Model (x1, x2) afterward
        """
        switch_stage = self.opts.switch_stage
        N = self.opts.N

        # Multi-phase OCP setup
        N_list=[switch_stage, 1, N - switch_stage] # 1 for the transition stage
        mp_ocp = AcadosMultiphaseOcp(N_list)

        # Create phases
        model_3d = self._create_vdp_3d_model()
        model_2d = self._create_vdp_2d_model()
        ocp_phase_0 = self._create_ocp(model_3d, first_stage=True)
        ocp_phase_1 = self._create_transition_ocp()
        ocp_phase_2 = self._create_ocp(model_2d)
        
        # Assign phases to OCP
        mp_ocp.set_phase(ocp_phase_0, 0)
        mp_ocp.set_phase(ocp_phase_1, 1)
        mp_ocp.set_phase(ocp_phase_2, 2)

        # Assign time steps
        step_sizes_list_with_transition = self.opts.step_size_list[:self.opts.switch_stage] + [1.0] + self.opts.step_size_list[self.opts.switch_stage:]
        mp_ocp.solver_options.time_steps = np.array(step_sizes_list_with_transition)
        mp_ocp.solver_options.tf = sum(step_sizes_list_with_transition)
        
        # Set solver options
        mp_ocp.solver_options.nlp_solver_type = self.opts.nlp_solver_type
        mp_ocp.solver_options.qp_solver = self.opts.qp_solver
        mp_ocp.mocp_opts.integrator_type = ['ERK', 'DISCRETE', 'ERK']

        self.mp_ocp = mp_ocp
    
    def _create_ocp(self, model, first_stage=False):
        """
        Defines the second-phase OCP using the 2D model (neglecting x3).
        """
        ocp = AcadosOcp()
        ocp.model = model

        nx = 3
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

        # constraints
        if first_stage:
            ocp.constraints.x0 = np.zeros((nx,1)) # initial state constraint 

        return ocp
    
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
    
    def _create_transition_model(self):
        model = AcadosModel()
        model.name = 'transition_model'
        x1 = ca.SX.sym("x1")
        x2 = ca.SX.sym("x2")
        x3 = ca.SX.sym("x3")
        model.x = ca.vertcat(x1, x2, x3)
        model.disc_dyn_expr = ca.vertcat(x1, x2)
        return model

    def _create_transition_ocp(self):
        ocp = AcadosOcp()
        ocp.model = self._create_transition_model() 
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.model.cost_y_expr = ca.vertcat(ocp.model.x[0], ocp.model.x[1])
        ocp.cost.W = np.diag([0.0, 0.0])
        ocp.cost.yref = np.array([0., 0.]) # the reference values are overwritten later
        return ocp

    def _create_sim_3d(self, json_file_suffix="vdp_3d_sim"):
        """
        Creates an AcadosSim solver for the 3D model (useful for closed-loop simulation).
        """
        model_3d = self._create_vdp_3d_model()
        sim = AcadosSim()
        sim.model = model_3d

        # Pick a step size for simulation (same as the first step size)
        sim.solver_options.T = self.opts.step_size_list[0]
        sim.solver_options.integrator_type = "ERK"
        
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
        N = len(self.opts.step_size_list)  # Horizon length without switching
        t_eval = t0  # Start from the given time

        offset = 0
        for stage in range(N):
            if stage == self.opts.switch_stage:
                offset += 1 # no penalty for switching stage, no reference update
            x1_ref, x2_ref = self.Phi_t(t_eval)  # Get reference at time t_eval
            y_ref = np.array([x1_ref, x2_ref, 0.0, 0.0])
            self.acados_ocp_solver.set(stage + offset, "yref", y_ref)
            t_eval += self.opts.step_size_list[stage]

        # Set terminal reference
        x1_ref, x2_ref = self.Phi_t(t_eval)
        y_ref_N = np.array([x1_ref, x2_ref])
        self.acados_ocp_solver.set(N+1, "yref", y_ref_N)

    def solve(self, x0, t0):
        """
        Solves the MPC problem:
        1. Sets the initial guess with x0.
        2. Sets the reference trajectory using self.Phi_t at time t.
        3. Solves the OCP.
        """
        # Set initial guess
        self.set_initial_state(x0)

        # Set reference trajectory
        self.set_reference_trajectory(t0)

        # Solve the OCP
        status = self.acados_ocp_solver.solve()
        
        if status != 0:
            print(f"[VanDerPolMPC] OCP solver returned status {status}.")

        # Return first control input
        return self.acados_ocp_solver.get(0, "u")

    def get_planned_trajectory(self):
        N = len(self.opts.step_size_list)
        traj_x, traj_u = [], []
        for i in range(N+1):  # N+1 due to transition stage
            x_i = self.acados_ocp_solver.get(i, "x")
            u_i = self.acados_ocp_solver.get(i, "u")
            traj_x.append(x_i)
            traj_u.append(u_i)
        x_N = self.acados_ocp_solver.get(N+1, "x")
        traj_x.append(x_N)
        return traj_x, traj_u
    
  
