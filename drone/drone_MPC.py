import numpy as np
import casadi as ca
import time
from dataclasses import dataclass, field
from typing import Optional, List

from acados_template import (
    AcadosModel,
    AcadosOcp,
    AcadosMultiphaseOcp,
    AcadosOcpSolver,
    AcadosSim,
    AcadosSimSolver,
)

###############################################################################
# Options Dataclass
###############################################################################

@dataclass
class DroneMPCOptions:
    """
    Holds relevant parameters for configuring the DroneMPC.
    """
    # Horizon & discretization
    N: int = 20
    switch_stage: int = 20
    step_sizes: List[float] = field(default_factory=lambda: [0.01]*20)

    # Solver / integrator settings
    nlp_solver_type: str = "SQP"
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM" # "PARTIAL_CONDENSING_OSQP" 
    integrator_type: str = "IRK"
    levenberg_marquardt: str = 8e-3

    # Physical parameters
    M: float = 2.0       # drone mass in kg
    m: float = 0.1       # load mass in kg (small compared to drone mass)
    Ixx: float = 0.05    # moment of inertia (kg·m²)
    g: float = 9.81      # gravitational acceleration (m/s²)
    c: float = 1.0       # rotor thrust constant (N per unit input)
    L_rot: float = 0.2   # half distance between rotors in m

    # Pendulum parameters (used only if pendulum is active)
    k: float = 100.0     # spring stiffness (N/m)
    l0: float = 0.2      # spring rest length in m

    # Cost weighting matrices
    #  - For ignoring-pendulum & direct-thrust phases (4D cost: [y, z, y_dd, z_dd]):
    #       W = blockdiag(Q, Q_acc)
    #  - For full-pendulum phases (4D cost: [y, z, y_load_dd, z_load_dd]):
    #       W = blockdiag(Q, Q_acc_load)
    #  - For transition (2D cost: [y_load_dd, z_load_dd]), we also use Q_acc_load
    Q: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0]))        # on [y, z]
    Q_acc: np.ndarray = field(default_factory=lambda: np.diag([0.05, 0.05, 0.00]))      # on [y_dd, z_dd]
    Q_acc_load: np.ndarray = field(default_factory=lambda: np.diag([0.1, 0.1])) # on [y_load_dd, z_load_dd]
    R_reg: np.ndarray = field(default_factory=lambda: np.diag([1e-3, 1e-3])) # very small penalty on inputs for regularization
    Q_trans: np.ndarray = field(default_factory=lambda: np.diag([0.03, 0.03]))

    # Constraints
    #  - Actuator model: states are [w1,w2], inputs are [dw1,dw2]
    #  - Direct thrust: inputs are [F1,F2]
    w_min: float = -10.0
    w_max: float = 10.0
    w_dot_min: float = -5.0
    w_dot_max: float = 5.0
    phi_min: float = -0.7
    phi_max: float = 0.7
    F_min: float = -15.0
    F_max: float = 15.0

    # Whether to create a sim solver for the full model
    create_sim: bool = True


###############################################################################
# Main Drone MPC Class
###############################################################################

class DroneMPC:

    def __init__(self, options: DroneMPCOptions):
        self.opts = options
        self.timestamp = int(time.time() * 1000)

        # Flag to indicate multi-phase usage
        self.multi_phase = False

        # Build the appropriate OCP based on switch_stage
        if self.opts.switch_stage == 0:
            # Single-phase, ignoring pendulum (but with actuator model).
            self._create_single_phase_ocp_ignore_pendulum()
        elif self.opts.switch_stage >= self.opts.N:
            # Single-phase, full pendulum + actuator
            self._create_single_phase_ocp_full_pendulum()
        else:
            # Multi-phase OCP
            self.multi_phase = True
            self._create_multi_phase_ocp()

        # Create the Acados solver
        json_file = f"acados_ocp_drone_{self.opts.switch_stage}_{self.timestamp}.json"
        self.acados_ocp_solver = AcadosOcpSolver(self.total_ocp, json_file=json_file)

        # Optionally create a single AcadosSimSolver for the full pendulum model only
        if self.opts.create_sim:
            self._create_sim_full_model()


    ###########################################################################
    # Public Methods
    ###########################################################################

    def solve(self, x0: np.ndarray, y_ref: Optional[np.ndarray] = None):
        """
        - Set initial state for the first phase.
        - If a reference is given, set it for all stages in the cost.
        - Solve the OCP and return the first control input.
        """

        if self.opts.switch_stage == 0:
            x0= self.to_8d(x0)

        # Phase 0 (index 0) gets x0 constraints
        self.acados_ocp_solver.set(0, "lbx", x0)
        self.acados_ocp_solver.set(0, "ubx", x0)

        # If reference is provided, set it (assuming y_ref = [y*, z*])
        if y_ref is not None:
            self._set_reference_in_cost(y_ref)

        status = self.acados_ocp_solver.solve()
        if status != 0:
            print(f"[DroneMPC] solver returned status {status}.")

        # Return the first control input
        return self.acados_ocp_solver.get(0, "u")

    def get_planned_trajectory(self):
        """
        Returns (traj_x, traj_u) for the entire horizon.
        If multi-phase, note there's an extra transition stage => total = N + 1.
        """
        N_total = self.opts.N
        if self.multi_phase:
            N_total += 1

        traj_x = []
        traj_u = []
        for i in range(N_total):
            x_i = self.acados_ocp_solver.get(i, "x")
            u_i = self.acados_ocp_solver.get(i, "u")
            traj_x.append(x_i)
            traj_u.append(u_i)

        x_end = self.acados_ocp_solver.get(N_total, "x")
        traj_x.append(x_end)
        return traj_x, traj_u
    
    def set_initial_guess(self, x0_full: np.ndarray, u_guess: np.ndarray):
        # Single-phase cases:
        if not self.multi_phase:
            # We have a horizon of N steps => N shooting intervals, so N+1 states
            N_horizon = self.opts.N
            if self.opts.switch_stage == 0:
                x0 = self.to_8d(x0_full)
            else:
                x0 = x0_full   
            for stage in range(N_horizon):
                self.acados_ocp_solver.set(stage, "x", x0)
                self.acados_ocp_solver.set(stage, "u", u_guess)
            self.acados_ocp_solver.set(N_horizon, "x", x0)
        else:
            N0 = self.opts.switch_stage       # horizon for phase 0
            N2 = self.opts.N - N0             # horizon for phase 2
            for stage in range(N0):
                self.acados_ocp_solver.set(stage, "x", x0_full)
                self.acados_ocp_solver.set(stage, "u", u_guess)

            # The discrete transition at stage N0: no input
            self.acados_ocp_solver.set(N0, "x", x0_full)

            x0_simpl = self.to_8d(x0_full)  # or to_8d if needed
            for stage in range(N0 + 1, N0 + 1 + N2):
                self.acados_ocp_solver.set(stage, "x", x0_simpl)
                self.acados_ocp_solver.set(stage, "u", u_guess)
            # terminal node
            self.acados_ocp_solver.set(N0 + 1 + N2, "x", x0_simpl)

    def to_8d(self, x_full):
        """Ignore pendulum + actuator => 8D."""
        return np.array([
            x_full[0],   # y
            x_full[1],   # z
            x_full[2],   # phi
            x_full[5],   # y_dot
            x_full[6],   # z_dot
            x_full[7],   # phi_dot
            x_full[10],  # w1
            x_full[11],  # w2
        ])

    ###########################################################################
    # Single-Phase OCPs
    ###########################################################################

    def _create_single_phase_ocp_ignore_pendulum(self):
        """
        Single-phase ignoring pendulum, but with actuator model.
        State: x = [y, z, phi, y_dot, z_dot, phi_dot, w1, w2]
        Control: u = [dw1, dw2]
        Cost: cost_expr = [ y, z, y_dd, z_dd ]  (dimension 4)
        Weighted by blockdiag(Q, Q_acc).
        """
        ocp = AcadosOcp()

        # Effective mass includes the small load mass
        M_eff = self.opts.M + self.opts.m
        model = self._build_drone_actuator_model(M_eff)
        ocp.model = model

        ocp.solver_options.N_horizon = self.opts.N
        ocp.solver_options.tf = sum(self.opts.step_sizes)

        # Nonlinear LS cost:
        cost_expr = self._cost_expr_drone_actuator_ignore_pendulum(model)  # dimension 4
        ocp.model.cost_y_expr = cost_expr
        ocp.cost.cost_type = "NONLINEAR_LS"

        # Terminal cost only on [y, z] => first 2 components
        ocp.model.cost_y_expr_e = cost_expr[0:2]
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        # Stage weighting
        # W_stage = np.block([
        #     [self.opts.Q,           np.zeros((2,2))],
        #     [np.zeros((2,2)),       self.opts.Q_acc]
        # ])
        W_stage = np.block([
            [self.opts.Q,             np.zeros((2, 3)),         np.zeros((2, 2))],
            [np.zeros((3, 2)),        self.opts.Q_acc,          np.zeros((3, 2))],
            [np.zeros((2, 2)),        np.zeros((2, 3)),          self.opts.R_reg]
        ])
        ocp.cost.W = W_stage
        ocp.cost.W_e = self.opts.Q

        ocp.cost.yref = np.zeros(7)   # [y, z, y_dd, z_dd]
        ocp.cost.yref_e = np.zeros(2) # [y, z]

        # Constraints
        nx = model.x.size()[0]
        ocp.constraints.x0 = np.zeros(nx)

        # phi => index 2
        ocp.constraints.idxbx = np.array([2])
        ocp.constraints.ubx = np.array([self.opts.phi_max])
        ocp.constraints.lbx = np.array([self.opts.phi_min])

        # w1,w2 => indices 6,7
        ocp.constraints.idxbx = np.append(ocp.constraints.idxbx, [6, 7])
        ocp.constraints.ubx = np.append(ocp.constraints.ubx, [self.opts.w_max, self.opts.w_max])
        ocp.constraints.lbx = np.append(ocp.constraints.lbx, [self.opts.w_min, self.opts.w_min])

        # controls => [dw1, dw2]
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.ubu = np.array([self.opts.w_dot_max, self.opts.w_dot_max])
        ocp.constraints.lbu = np.array([self.opts.w_dot_min, self.opts.w_dot_min])

        self._set_ocp_solver_options(ocp)
        self.total_ocp = ocp


    def _create_single_phase_ocp_full_pendulum(self):
        """
        Single-phase with full pendulum + actuator model.
        State: x = 12D
        Control: u = [dw1, dw2]
        Cost: cost_expr = [ y, z, y_load_dd, z_load_dd ]
        Weighted by blockdiag(Q, Q_acc_load).
        Terminal cost only on [y, z].
        """
        ocp = AcadosOcp()

        # Build the full pendulum model from external ODE file
        model_full = self._build_full_pendulum_model()
        ocp.model = model_full

        ocp.solver_options.N_horizon = self.opts.N
        ocp.solver_options.tf = sum(self.opts.step_sizes)

        # Nonlinear LS cost
        cost_expr = self._cost_expr_full_pendulum(model_full)  # dimension 4
        ocp.model.cost_y_expr = cost_expr
        ocp.cost.cost_type = "NONLINEAR_LS"

        # Terminal cost on first 2 comps => [y, z]
        ocp.model.cost_y_expr_e = ca.vertcat(cost_expr[0], cost_expr[1])
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        W_stage = np.block([
            [self.opts.Q,             np.zeros((2, 2)),         np.zeros((2, 2))],
            [np.zeros((2, 2)),        self.opts.Q_acc_load,     np.zeros((2, 2))],
            [np.zeros((2, 2)),        np.zeros((2, 2)),         self.opts.R_reg]
        ])
        ocp.cost.W = W_stage
        ocp.cost.W_e = self.opts.Q

        ocp.cost.yref = np.zeros(6)
        ocp.cost.yref_e = np.zeros(2)

        # store stage cost function with full model, this is used for evaluating the costs in closed loop simulation
        yref_ca = ca.MX.sym("yref", 6)

        diff = cost_expr - yref_ca
        stage_cost_sym = diff.T @ W_stage @ diff

        # 4) Create a CasADi function
        self.stage_cost_func_full = ca.Function(
            "stage_cost_func_full",
            [ocp.model.x, ocp.model.u, yref_ca],
            [stage_cost_sym],
            ["x", "u", "y_ref"],
            ["stage_cost"]
        )

        # Constraints
        nx = model_full.x.size()[0]
        ocp.constraints.x0 = np.zeros(nx)

        # phi => index 2
        ocp.constraints.idxbx = np.array([2])
        ocp.constraints.ubx = np.array([self.opts.phi_max])
        ocp.constraints.lbx = np.array([self.opts.phi_min])

        # w1,w2 => indices 10,11
        ocp.constraints.idxbx = np.append(ocp.constraints.idxbx, [10, 11])
        ocp.constraints.ubx = np.append(ocp.constraints.ubx, [self.opts.w_max, self.opts.w_max])
        ocp.constraints.lbx = np.append(ocp.constraints.lbx, [self.opts.w_min, self.opts.w_min])

        # control => [dw1, dw2]
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.ubu = np.array([self.opts.w_dot_max, self.opts.w_dot_max])
        ocp.constraints.lbu = np.array([self.opts.w_dot_min, self.opts.w_dot_min])

        self._set_ocp_solver_options(ocp)
        self.total_ocp = ocp


    ###########################################################################
    # Multi-Phase OCP
    ###########################################################################

    def _create_multi_phase_ocp(self):
        """
        Phase 0: Full pendulum + actuator (no terminal cost).
        Phase 1: 1-step discrete transition, cost = only [y_load_dd, z_load_dd].
        Phase 2: Simplified direct-thrust model, terminal cost on [y, z].
        Only phase 0 sets x0 constraints.
        """
        # Partition
        N0 = self.opts.switch_stage
        N2 = self.opts.N - N0

        mp_ocp = AcadosMultiphaseOcp([N0, 1, N2])

        # ------- Phase 0: Full pendulum + actuator, no terminal cost -------
        ocp_0 = AcadosOcp()
        model_0 = self._build_full_pendulum_model()  # 12D
        ocp_0.model = model_0

        # Nonlinear LS cost: dimension=4
        cost_expr_0 = self._cost_expr_full_pendulum(model_0)
        ocp_0.model.cost_y_expr = cost_expr_0
        ocp_0.cost.cost_type = "NONLINEAR_LS"

        # No terminal cost for phase 0 (we simply don't define cost_y_expr_e).
        W_0 = np.block([
            [self.opts.Q,             np.zeros((2, 2)),         np.zeros((2, 2))],
            [np.zeros((2, 2)),        self.opts.Q_acc_load,     np.zeros((2, 2))],
            [np.zeros((2, 2)),        np.zeros((2, 2)),         self.opts.R_reg]
        ])
        ocp_0.cost.W = W_0
        ocp_0.cost.yref = np.zeros(6)

        nx_0 = model_0.x.size()[0]
        ocp_0.constraints.x0 = np.zeros(nx_0)  # initial state only in phase 0

        # phi => index=2
        ocp_0.constraints.idxbx = np.array([2])
        ocp_0.constraints.ubx = np.array([self.opts.phi_max])
        ocp_0.constraints.lbx = np.array([self.opts.phi_min])
        # w => index=10,11
        ocp_0.constraints.idxbx = np.append(ocp_0.constraints.idxbx, [10, 11])
        ocp_0.constraints.ubx = np.append(ocp_0.constraints.ubx, [self.opts.w_max, self.opts.w_max])
        ocp_0.constraints.lbx = np.append(ocp_0.constraints.lbx, [self.opts.w_min, self.opts.w_min])
        # u => [dw1, dw2]
        ocp_0.constraints.idxbu = np.array([0, 1])
        ocp_0.constraints.ubu = np.array([self.opts.w_dot_max, self.opts.w_dot_max])
        ocp_0.constraints.lbu = np.array([self.opts.w_dot_min, self.opts.w_dot_min])

        self._set_ocp_solver_options(ocp_0)

        # ------- Phase 1: discrete transition, cost = only load acceleration -------
        ocp_1 = AcadosOcp()
        transition_model = self._build_transition_model()  # input is 12D
        ocp_1.model = transition_model

        # Reuse the load acceleration expressions from model_0.
        # We stored them as a CasADi Function in model_0.load_acc_func(...).
        # So cost_y_expr = model_0.load_acc_func( x_in ), where x_in is 12D.
        # But we can't store that expression directly in transition_model itself,
        # so we do it after we create the OCP:
        # load_acc_expr = model_0.load_acc_func(transition_model.x)
        transition_costs_expr = self._cost_expr_full_transition(transition_model)
        ocp_1.model.cost_y_expr = transition_costs_expr
        ocp_1.cost.cost_type = "NONLINEAR_LS"
        ocp_1.cost.W = self.opts.Q_trans
        ocp_1.cost.yref = np.zeros(2) # dimension=2 => [y_load_dd, z_load_dd]

        self._set_ocp_solver_options(ocp_1)

        # ------- Phase 2: simplified direct-thrust model, terminal cost on [y, z] -------
        ocp_2 = AcadosOcp()
        model_2 = self._build_drone_actuator_model(M_eff=self.opts.M + self.opts.m)
        ocp_2.model = model_2

        cost_expr_2 = self._cost_expr_drone_actuator_ignore_pendulum(model_2)  # dimension=4
        ocp_2.model.cost_y_expr = cost_expr_2
        ocp_2.cost.cost_type = "NONLINEAR_LS"

        # Terminal cost on first 2 comps => [y, z]
        ocp_2.model.cost_y_expr_e = ca.vertcat(cost_expr_2[0], cost_expr_2[1])
        ocp_2.cost.cost_type_e = "NONLINEAR_LS"

        # W_2 = np.block([
        #     [self.opts.Q,       np.zeros((2,2))],
        #     [np.zeros((2,2)),   self.opts.Q_acc]
        # ])
        W_2 = np.block([
            [self.opts.Q,             np.zeros((2, 3)),         np.zeros((2, 2))],
            [np.zeros((3, 2)),        self.opts.Q_acc,          np.zeros((3, 2))],
            [np.zeros((2, 2)),        np.zeros((2, 3)),          self.opts.R_reg]
        ])
        ocp_2.cost.W = W_2
        ocp_2.cost.W_e = self.opts.Q
        ocp_2.cost.yref = np.zeros(7)
        ocp_2.cost.yref_e = np.zeros(2)

        # no x0 for phase 2
        nx_2 = model_2.x.size()[0]
        ocp_2.constraints.x0 = np.zeros(nx_2)

        # phi => index=2
        ocp_2.constraints.idxbx = np.array([2])
        ocp_2.constraints.ubx = np.array([self.opts.phi_max])
        ocp_2.constraints.lbx = np.array([self.opts.phi_min])
        # w => index 6, 7
        ocp_2.constraints.idxbx = np.append(ocp_2.constraints.idxbx, [6, 7])
        ocp_2.constraints.ubx = np.append(ocp_2.constraints.ubx, [self.opts.w_max, self.opts.w_max])
        ocp_2.constraints.lbx = np.append(ocp_2.constraints.lbx, [self.opts.w_min, self.opts.w_min])
        # u => [dw1, dw2]
        ocp_2.constraints.idxbu = np.array([0, 1])
        ocp_2.constraints.ubu = np.array([self.opts.w_dot_max, self.opts.w_dot_max])
        ocp_2.constraints.lbu = np.array([self.opts.w_dot_min, self.opts.w_dot_min])

        self._set_ocp_solver_options(ocp_2)

        # Combine phases
        mp_ocp.set_phase(ocp_0, 0)
        mp_ocp.set_phase(ocp_1, 1)
        mp_ocp.set_phase(ocp_2, 2)

        self._set_ocp_solver_options(mp_ocp)

        # Overwrite some solver options that need to change for multi phase problems
        # Time steps (including discrete transition)
        step_sizes_list = list(self.opts.step_sizes[:N0]) + [1.0] + list(self.opts.step_sizes[N0:])
        mp_ocp.solver_options.time_steps = np.array(step_sizes_list)
        mp_ocp.solver_options.tf = sum(step_sizes_list)

        # Integrator choices
        mp_ocp.mocp_opts.integrator_type = [
            self.opts.integrator_type,  # phase 0: full pendulum
            "DISCRETE",                 # phase 1: transition
            "ERK",                      # phase 2: direct thrust
        ]

        self.total_ocp = mp_ocp


    ###########################################################################
    # Setting References 
    ###########################################################################

    def _set_reference_in_cost(self, pos_ref: np.ndarray):
        """
        Similar to how references are handled in vdp. We assume y_ref = [y_target, z_target].
        We'll set [y_ref, z_ref, 0, 0] for 4D costs, and [y_ref, z_ref] for 2D costs.
        """
        N = self.opts.N
        y_ref, z_ref = pos_ref[0], pos_ref[1]

        # For each shooting node: set a suitable yref dimension
        offset = 0
        for stage in range(N):
            if stage != 0 and stage == self.opts.switch_stage: # 0 stage means only 2d model and no switch
                offset += 1 # no reference update for witching stage, penalize large x3
                continue
            elif stage >= self.opts.switch_stage:
                y_ref_acados = np.array([y_ref, z_ref, 0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                y_ref_acados = np.array([y_ref, z_ref, 0.0, 0.0, 0.0, 0.0])
            self.acados_ocp_solver.set(stage + offset, "yref", y_ref_acados)     
        y_ref_N_acados = np.array([y_ref, z_ref])
        self.acados_ocp_solver.set(N+offset, "yref", y_ref_N_acados)
            

    ###########################################################################
    # Model-Building Routines
    ###########################################################################

    def _build_drone_actuator_model(self, M_eff: float) -> AcadosModel:
        """
        2D drone with actuator model, ignoring pendulum:
         State = [y, z, phi, y_dot, z_dot, phi_dot, w1, w2]
         Control = [dw1, dw2]
         ODE:
           y_dd = -(w1 + w2)* c * sin(phi) / M_eff
           z_dd = ( (w1 + w2)* c * cos(phi) - M_eff*g ) / M_eff
           phi_dd= (L_rot*c / Ixx)*(w1 - w2)
           w1_dot= dw1
           w2_dot= dw2
        """
        x = ca.SX.sym("x", 8)
        u = ca.SX.sym("u", 2)

        y, z, phi = x[0], x[1], x[2]
        y_dot, z_dot, phi_dot = x[3], x[4], x[5]
        w1, w2 = x[6], x[7]
        dw1, dw2 = u[0], u[1]

        c_ = self.opts.c
        g_ = self.opts.g
        L_ = self.opts.L_rot
        I_ = self.opts.Ixx

        T = (w1 + w2)*c_
        y_dd = -T*ca.sin(phi)/M_eff
        z_dd = (T*ca.cos(phi) - M_eff*g_)/M_eff
        phi_dd = (L_*c_ / I_)*(-w1 + w2)

        xdot = ca.vertcat(
            y_dot,
            z_dot,
            phi_dot,
            y_dd,
            z_dd,
            phi_dd,
            dw1,
            dw2
        )

        model = AcadosModel()
        model.name = f"drone_ignorepend_{self.timestamp}"
        model.x = x
        model.u = u
        model.xdot = ca.SX.sym("xdot", 8)
        model.f_expl_expr = xdot
        model.f_impl_expr = model.xdot - model.f_expl_expr
        return model

    def _build_full_pendulum_model(self) -> AcadosModel:
        """
        Full 12D model loaded from your ODE file eom_2d_quadro_springpend_explicit_casadi.py:
           x = [y, z, phi, r, theta, y_dot, z_dot, phi_dot, r_dot, theta_dot, w1, w2]
           u = [dw1, dw2]
         We'll also define a CasADi function load_acc_func(x_12) => [y_load_dd, z_load_dd],
         which we can reuse in the transition phase cost.
        """
        import eom_2d_quadro_springpend_explicit_casadi as eomfile

        x = ca.MX.sym("x", 12)
        u = ca.MX.sym("u", 2)

        # Build the params vector for the ODE function (external forces = 0)
        params = ca.vertcat(
            self.opts.M,          # M
            self.opts.m,          # load mass
            self.opts.Ixx,        # Ixx
            self.opts.g,          # g
            self.opts.k,          # spring stiffness
            self.opts.l0,         # rest length
            self.opts.c,          # rotor thrust constant
            self.opts.L_rot,      # half rotor distance
            0.0, 0.0, 0.0, 0.0    # external forces set to zero
        )

        # ODE from file
        x_dot_expr = eomfile.eom_2d_quadro_springpend_explicit(x, u, params)

        # Create the AcadosModel
        model = AcadosModel()
        model.name = f"drone_full_{self.timestamp}"
        model.x = x
        model.u = u
        nx = x.shape[0]
        model.xdot = ca.MX.sym("xdot", nx)
        model.f_expl_expr = x_dot_expr
        model.f_impl_expr = model.xdot - model.f_expl_expr

        # ----------------------------------------------------------
        # Build a CasADi expression for the load acceleration:
        #   y_load = y + r*sin(theta)
        #   z_load = z - r*cos(theta)
        # then differentiate twice w.r.t. time
        y_drone = x[0]
        z_drone = x[1]
        r_val   = x[3]
        th_val  = x[4]

        y_load = y_drone + r_val*ca.sin(th_val)
        z_load = z_drone - r_val*ca.cos(th_val)

        # first derivative
        dy_load_dx = ca.jacobian(y_load, x)
        y_load_dot = dy_load_dx @ x_dot_expr
        dz_load_dx = ca.jacobian(z_load, x)
        z_load_dot = dz_load_dx @ x_dot_expr

        # second derivative
        dy_load_dot_dx = ca.jacobian(y_load_dot, x)
        y_load_ddot = dy_load_dot_dx @ x_dot_expr
        dz_load_dot_dx = ca.jacobian(z_load_dot, x)
        z_load_ddot = dz_load_dot_dx @ x_dot_expr

        # Store as a Function => model.load_acc_func( x_12 ) => [y_load_dd, z_load_dd]
        # We'll need this for the transition cost.
        self.load_acc_func = ca.Function(
            "load_acc_func",
            [model.x, model.u], [y_load_ddot, z_load_ddot],
            ["x_in", "u_in"], ["y_load_dd", "z_load_dd"]
        )

        return model

    def _build_transition_model(self) -> AcadosModel:
        """
        Discrete map from 12D => 6D: [y, z, phi, y_dot, z_dot, phi_dot].
        Input state = x_in (12D).
        """
        x_in = ca.SX.sym("x_in", 12)
        disc_expr = ca.vertcat(
            x_in[0],  # y
            x_in[1],  # z
            x_in[2],  # phi
            x_in[5],  # y_dot
            x_in[6],  # z_dot
            x_in[7],  # phi_dot
            x_in[9],  # w_1
            x_in[10], # w_2
        )

        model = AcadosModel()
        model.name = f"transition_{self.timestamp}"
        model.x = x_in
        model.disc_dyn_expr = disc_expr
        return model

    ###########################################################################
    # Cost Expressions
    ###########################################################################

    def _cost_expr_drone_actuator_ignore_pendulum(self, model: AcadosModel) -> ca.SX:
        """
        For ignoring-pendulum with actuator states:
         cost_y_expr = [ y, z, y_dd, z_dd ]
         - State: x=[y,z,phi,y_dot,z_dot,phi_dot,w1,w2]
         - xdot=[y_dot,z_dot,phi_dot,y_dd,z_dd,phi_dd,w1_dot,w2_dot]
           => y_dd=xdot[3], z_dd=xdot[4]
        """
        xdot_expr = model.f_expl_expr
        y_expr = model.x[0]
        z_expr = model.x[1]
        y_dd_expr = xdot_expr[3]
        z_dd_expr = xdot_expr[4]
        phi_dd_expr = xdot_expr[5]
        u_1 = model.u[0]
        u_2 = model.u[1]
        cost_expr = ca.vertcat(y_expr, z_expr, y_dd_expr, z_dd_expr, phi_dd_expr, u_1, u_2)
        return cost_expr

    def _cost_expr_full_pendulum(self, model: AcadosModel) -> ca.SX:
        """
        cost_y_expr = [ y, z, y_load_dd, z_load_dd ]
        We'll reuse the symbolic function that we stored in 'self.load_acc_func':
           [y_load_dd, z_load_dd] = self.load_acc_func(x).
        """
        x_sym = model.x
        y_drone = x_sym[0]
        z_drone = x_sym[1]

        # compute [y_load_ddot, z_load_ddot] using the stored function
        load_acc = self.load_acc_func(x_sym, model.u)  # => [y_load_dd, z_load_dd]
        y_load_ddot = load_acc[0]
        z_load_ddot = load_acc[1]

        u_1 = model.u[0]
        u_2 = model.u[1]

        cost_expr = ca.vertcat(y_drone, z_drone, y_load_ddot, z_load_ddot, u_1, u_2)
        return cost_expr
    
    def _cost_expr_full_transition(self, model: AcadosModel) -> ca.SX:
        """
        cost_y_expr = [ y, z, y_load_dd, z_load_dd ]
        We'll reuse the symbolic function that we stored in 'self.load_acc_func':
           [y_load_dd, z_load_dd] = self.load_acc_func(x).
        """
        x_sym = model.x

        # compute [y_load_ddot, z_load_ddot] using the stored function
        load_acc = self.load_acc_func(x_sym, np.zeros(2))  # => [y_load_dd, z_load_dd]
        y_load_ddot = load_acc[0]
        z_load_ddot = load_acc[1]

        cost_expr = ca.vertcat(y_load_ddot, z_load_ddot)
        return cost_expr


    ###########################################################################
    # Solver Options
    ###########################################################################

    def _set_ocp_solver_options(self, ocp: AcadosOcp):
        ocp.solver_options.nlp_solver_type = self.opts.nlp_solver_type
        ocp.solver_options.qp_solver = self.opts.qp_solver
        ocp.solver_options.integrator_type = self.opts.integrator_type
        ocp.solver_options.nlp_solver_max_iter = 300
        ocp.solver_options.tol = 1e-6
        ocp.solver_options.qp_tol = 1e-2*ocp.solver_options.tol
        ocp.solver_options.print_level = 1
        ocp.solver_options.qp_solver_iter_max = 500
        ocp.solver_options.hessian_approx = "EXACT"#"GAUSS_NEWTON"
        ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        ocp.solver_options.levenberg_marquardt = self.opts.levenberg_marquardt # regularize Hessian stongly due to conditioning
        ocp.solver_options.hpipm_mode = "ROBUST" # problem seems a bit ill conditioned, try robust mode
        # ocp.solver_options.tf or time_steps must be set outside as appropriate.

    ###########################################################################
    # AcadosSimSolver for the Full Model Only
    ###########################################################################

    def _create_sim_full_model(self):
        """
        Create a single AcadosSimSolver for the full pendulum model for potential
        closed-loop simulation or debugging.
        """
        sim_full = AcadosSim()
        model_full = self._build_full_pendulum_model()
        sim_full.model = model_full

        sim_full.solver_options.T = self.opts.step_sizes[0]
        sim_full.solver_options.integrator_type = self.opts.integrator_type

        sim_solver = AcadosSimSolver(
            sim_full, json_file=f"acados_sim_full_{self.timestamp}.json"
        )
        self.sim_solver_full = sim_solver
