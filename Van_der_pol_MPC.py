from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver, AcadosMultiphaseOcp
import casadi as ca
import time

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
    step_size_list: List[float] = field(default_factory=lambda: [0.02] * 40)
    N: int = 40
    switch_stage: int = 20
    nlp_solver_type: str = "SQP"
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    integrator_type: str = "IRK"

    # Cost Weights
    Q_2d: np.ndarray = field(default_factory=lambda: np.diag([10.0, 10.0]))
    R_2d: np.ndarray = field(default_factory=lambda: np.diag([0.002, 0.002]))

    # Elliptical Constraints (x1, x2 centers & half-axis lengths a and b)
    ellipse_centers: Optional[np.ndarray] = None  # Shape (num_ellipses, 2)
    ellipse_half_axes: Optional[np.ndarray] = None  # Shape (num_ellipses, 2), where each row is [a, b]

   
class VanDerPolMPC:

    def __init__(self, options: VanDerPolMPCOptions, Phi_t, create_sim=True):
        """
        Initializes the MPC problem using the provided options.
        """
        self.opts = options  # Store options as self.opts
        self.Phi_t = Phi_t

        # time stamp to avoid solvers interfering with each other
        self.timestamp = int(time.time()*1000)

        self.multi_phase = False
        if self.opts.switch_stage == 0:
            # Use only 2d model
            self._create_single_phase_ocp(model_dim=2)
        elif self.opts.switch_stage >= self.opts.N:
            # Use only 3d model
            self._create_single_phase_ocp(model_dim=3)
        else:
            # Create a 2-phase OCP (3D → 2D transition)
            self.multi_phase = True
            self._create_multiphase_ocp()

        # Create Acados Solver
        json_file = f"acados_ocp_vdp_{self.opts.switch_stage}_{self.timestamp}.json"
        self.acados_ocp_solver = AcadosOcpSolver(self.total_ocp, json_file=json_file)

        # if self.opts.ellipse_centers is not None and self.opts.ellipse_half_axes is not None:
        #     # This removes the ellipsoidal constraint from the first stage, as it can cause problems in erroneous sim.
        #     # It is done such that the x0 is not infeasible. Acados python interface does not support different
        #     # constraints for different stages, it needs to be done in this hacky way.
        #     self.remove_constraints_from_first_stage()

        # Create Acados Simulation Solver (for closed-loop simulation)
        if create_sim:
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
        step_sizes_list_with_transition = list(self.opts.step_size_list[:self.opts.switch_stage]) + [1.0] + list(self.opts.step_size_list[self.opts.switch_stage:])
        mp_ocp.solver_options.time_steps = np.array(step_sizes_list_with_transition)
        mp_ocp.solver_options.tf = sum(step_sizes_list_with_transition)
        
        # Set solver options
        mp_ocp.solver_options.nlp_solver_type = self.opts.nlp_solver_type
        mp_ocp.solver_options.qp_solver = self.opts.qp_solver
        mp_ocp.mocp_opts.integrator_type = [self.opts.integrator_type, 'DISCRETE', 'ERK']

        self.total_ocp = mp_ocp

    def _create_single_phase_ocp(self, model_dim):
        if model_dim == 2:
            # 2d model case
            model = self._create_vdp_2d_model()
        else:
            # 3d model case
            model = self._create_vdp_3d_model()
        
        ocp = self._create_ocp(model, first_stage=True)
        ocp.solver_options.time_steps = np.array(self.opts.step_size_list)
        ocp.solver_options.tf = sum(self.opts.step_size_list)
        
        # Set solver options
        ocp.solver_options.nlp_solver_type = self.opts.nlp_solver_type
        ocp.solver_options.qp_solver = self.opts.qp_solver
        ocp.solver_options.integrator_type = self.opts.integrator_type
        ocp.solver_options.N_horizon = self.opts.N
        #ocp.solver_options.nlp_solver_warm_start_first_qp = False


        self.total_ocp = ocp
    
    def _create_ocp(self, model, first_stage=False):
        """
        Defines the second-phase OCP using the 2D model (neglecting x3).
        """
        ocp = AcadosOcp()
        ocp.model = model

        if self.opts.switch_stage == 0:
            nx = 2
        else:
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
        # Apply Circular Constraints if Provided
        if self.opts.ellipse_centers is not None and self.opts.ellipse_half_axes is not None:
            ocp = self._add_region_constraints_to_ocp(ocp)

        # box constraints for u, necessary to prevent jittering
        u_min = np.array([-10.0, -10.0])  # Example: Min control limits
        u_max = np.array([10.0, 10.0])    # Example: Max control limits

        # Apply box constraints on control inputs
        ocp.constraints.lbu = u_min
        ocp.constraints.ubu = u_max
        ocp.constraints.idxbu = np.array([0, 1])  # Apply to both control inputs

        return ocp

    def _add_region_constraints_to_ocp(self, ocp):
        x1, x2 = ocp.model.x[0], ocp.model.x[1]
        num_ellipses = self.opts.ellipse_centers.shape[0]
        h_expr_list = []

        for i in range(num_ellipses):
            x1_c, x2_c = self.opts.ellipse_centers[i, :]
            a, b = self.opts.ellipse_half_axes[i, :]
            h_expr = ((x1 - x1_c) / a) ** 2 + ((x2 - x2_c) / b) ** 2 - 1
            h_expr_list.append(h_expr)

        h_expr_full = ca.vertcat(*h_expr_list)

        ocp.model.con_h_expr = h_expr_full
        ocp.dims.nh = len(h_expr_list)
        ocp.constraints.lh = np.zeros(len(h_expr_list))
        ocp.constraints.uh = np.full(len(h_expr_list), 1e15)

        ocp.model.con_h_expr_e = h_expr_full
        ocp.dims.nh_e = len(h_expr_list)
        ocp.constraints.lh_e = np.zeros(len(h_expr_list))
        ocp.constraints.uh_e = np.full(len(h_expr_list), 1e15)
        # ocp.constraints.idxsh = np.arange(len(h_expr_list))
        # ocp.cost.Zl = np.full(len(h_expr_list), 4.0)
        # ocp.cost.Zu = np.full(len(h_expr_list), 4.0)
        # ocp.cost.zl = np.full(len(h_expr_list), 2.0)
        # ocp.cost.zu = np.full(len(h_expr_list), 2.0)
        
        return ocp
    
    # def remove_constraints_from_first_stage(self):
    #     """
    #     Removes the elliptical constraints from stage 0 by setting very small lower bounds 
    #     and very large upper bounds to effectively deactivate them.
    #     """
    #     # Get number of constraints
    #     num_constraints = self.opts.ellipse_centers.shape[0]

    #     # Remove constraints at stage 0 by setting loose bounds
    #     self.acados_ocp_solver.constraints_set(0, "lh", np.full(num_constraints, -1e15))  # Large negative bound
    #     self.acados_ocp_solver.constraints_set(0, "uh", np.full(num_constraints, 1e15))   # Large positive bound

    
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
        xdot = ca.SX.sym("xdot", 3)

        u1 = ca.SX.sym("u1")  
        u2 = ca.SX.sym("u2")  
        u = ca.vertcat(u1, u2)

        # Define dynamics
        dx1 = x2 + self.opts.d1 * u1 + self.opts.epsilon_1 * x3
        dx2 = self.opts.mu * (1 - x1**2) * x2 - x1 + self.opts.epsilon_2 * x2 + self.opts.epsilon_3 * x3 + self.opts.d2 * u2
        dx3 = - (1.0 / self.opts.epsilon_4) * x3 + self.opts.d3 * u2

        f_expl = ca.vertcat(dx1, dx2, dx3)

        model = AcadosModel()
        model.name = f"vdp_3d_{self.timestamp}"
        model.x = x
        model.xdot = xdot
        model.u = u
        model.f_expl_expr = f_expl
        model.f_impl_expr = xdot - f_expl  # Implicit form

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
        xdot = ca.SX.sym("xdot", 2)

        u1 = ca.SX.sym("u1")  
        u2 = ca.SX.sym("u2")  
        u = ca.vertcat(u1, u2)

        dx1 = x2 + self.opts.d1 * u1
        dx2 = self.opts.mu * (1 - x1**2) * x2 - x1 + self.opts.epsilon_2 * x2 + self.opts.d2 * u2

        f_expl = ca.vertcat(dx1, dx2)

        model = AcadosModel()
        model.name = f"vdp_2d_{self.timestamp}"
        model.x = x
        model.xdot = xdot
        model.u = u
        model.f_expl_expr = f_expl
        model.f_impl_expr = xdot - f_expl

        return model
    
    def _create_transition_model(self):
        model = AcadosModel()
        model.name = f'transition_model_{self.timestamp}'
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
        ocp.model.cost_y_expr = ca.vertcat(ocp.model.x[2])
        ocp.cost.W = np.diag([20.0])
        ocp.cost.yref = np.array([0.]) # the reference values are overwritten later
        # Apply Circular Constraints if Provided
        if self.opts.ellipse_centers is not None and self.opts.ellipse_half_axes is not None:
            ocp = self._add_region_constraints_to_ocp(ocp)
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
        sim.solver_options.integrator_type = self.opts.integrator_type
        
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
        N = self.opts.N  # Horizon length without switching
        t_eval = t0  # Start from the given time

        offset = 0
        for stage in range(N):
            if stage is not 0 and stage == self.opts.switch_stage: # 0 stage means only 2d model and no switch
                offset += 1 # no reference update for witching stage, penalize large x3
            x1_ref, x2_ref = self.Phi_t(t_eval)  # Get reference at time t_eval
            y_ref = np.array([x1_ref, x2_ref, 0.0, 0.0])
            self.acados_ocp_solver.set(stage + offset, "yref", y_ref)
            t_eval += self.opts.step_size_list[stage]

        # Set terminal reference
        x1_ref, x2_ref = self.Phi_t(t_eval)
        y_ref_N = np.array([x1_ref, x2_ref])
        self.acados_ocp_solver.set(N+offset, "yref", y_ref_N)

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
        N = self.opts.N
        if self.multi_phase:
            N += 1 # in multi-phase case, we need to account for the switching stage
        traj_x, traj_u = [], []
        for i in range(N):  # N+1 due to transition stage
            x_i = self.acados_ocp_solver.get(i, "x")
            u_i = self.acados_ocp_solver.get(i, "u")
            traj_x.append(x_i)
            traj_u.append(u_i)
        x_N = self.acados_ocp_solver.get(N, "x")
        traj_x.append(x_N)
        return traj_x, traj_u
    
  