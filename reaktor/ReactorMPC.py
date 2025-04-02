import numpy as np
import casadi as ca
import time
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

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
class ReactorMPCOptions:
    # Horizon & discretization
    N: int = 20                    # Total horizon (number of shooting intervals)
    switch_stage: int = 10         # Number of intervals to run full model (Phase 0); must be between 0 and N.
    step_sizes: List[float] = field(default_factory=lambda: [0.1]*20)

    # Solver settings
    nlp_solver_type: str = "SQP"
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    integrator_type: str = "IRK"
    levenberg_marquardt: float = 1e-3

    # Process parameters (for the chemical reactor)
    F: float = 5.0               # Flow rate
    V: float = 1.0               # Reactor volume
    T0: float = 300.0            # Feed temperature
    k0: float = 8.46e6           # Pre-exponential constant
    E: float = 5e4               # Activation energy
    R_const: float = 8.314       # Universal gas constant
    # Additional parameters (e.g., heat capacity, density, reaction enthalpy)
    delta_H: float = -20.0       # Reaction enthalpy [kJ/kmol] (example)
    rho: float = 20.0            # Density [kg/m^3] (example)
    cp: float = 0.02             # Heat capacity [kJ/kg/K] (example)

    # steady state values
    Q_s: float = 0
    T_s: float = 401.87
    CA_s: float = 1.95
    CA0_s: float = 4 

    # Cost weights
    # For Phase 0: economic cost weight and quadratic penalties on inputs
    Q_econ: float = 1.0         # Weight on the economic cost term (production rate)
    R_reactant: float = 1.0     # Weight on input costs for reactant
    R_heating: float = 1.0      # Weight on input costs for heating
    # For Phase 1 (transition), weight on deviation of T from T_s
    Q_trans: float = 1.0       # penalty on T - T_manifold in transition

    # Region constraints on the relative states (keep state in a region around the unstable steady state)
    CA_rel_min: float = -1.0    # Lower bound on CA - CA_s
    CA_rel_max: float =  1.0    # Upper bound on CA - CA_s
    T_rel_min: float = -30.0    # Lower bound on T - T_s
    T_rel_max: float =  30.0    # Upper bound on T - T_s

    # Input constraints (absolute)
    CA0_min: float = 0.0        # Cannot pull reactant; feed must be ≥ 0
    CA0_max: float = 10.0        # Upper bound for reactant feed
    Q_min: float = -15.0        # Heating lower bound (could be cooling if negative)
    Q_max: float = 15.0         # Heating upper bound

    # Whether to create a sim solver for the full model (for debugging or closed-loop simulation)
    create_sim: bool = True

    # Option for polynomial approximated of the algebraic expression
    poly_order: int = 3
    poly_fit_range: float = 0.1  # sample CA_rel in [-0.1, 0.1]


###############################################################################
# Mixed Chemical MPC Class
###############################################################################

class ReactorMPC:
    """
    Mixed-model MPC for a chemical reactor example.
    The MPC uses a full model (Phase 0) with dynamics written in relative terms ([CA-CA_s, T-T_s])
    and an economic cost (negative production rate) plus quadratic penalties on absolute inputs.
    A transition phase (Phase 1) maps the full state to only the relative concentration and penalizes deviations of T.
    In Phase 2 the fast dynamics are neglected (T is fixed to T_s), and only the slow dynamics (of CA) remain.
    State constraints ensure the deviation stays within a region around the unstable steady state.
    """
    def __init__(self, options: ReactorMPCOptions):
        self.opts = options
        self.timestamp = int(time.time() * 1000)
        self.multi_phase = False

        # Decide on the problem formulation based on switch_stage:
        if self.opts.switch_stage == 0:
            ValueError("switch_stage 0 not yet implemented for this example")
        elif self.opts.switch_stage >= self.opts.N:
            # Use full model for entire horizon (Phase 0 only)
            self.total_ocp = self._create_single_phase_ocp_full()
        else:
            # Mixed model: Phase 0 (full), Phase 1 (transition), Phase 2 (reduced)
            self.multi_phase = True

            # Call polynomial fitting for the reduced model manifold:
            self._fit_algebraic_polynomial()
            self.total_ocp = self._create_multi_phase_ocp()

        # Create the Acados solver
        json_file = f"acados_ocp_mixed_{self.opts.switch_stage}_{self.timestamp}.json"
        self.acados_ocp_solver = AcadosOcpSolver(self.total_ocp, json_file=json_file)

        # Optionally, create a simulation solver for the full model (Phase 0)
        if self.opts.create_sim:
            self._create_sim_full_model()

    ###########################################################################
    # Public Methods
    ###########################################################################

    def set_initial_state(self, x0_full: np.ndarray):
        """
        Set the initial state (in full model coordinates: [CA-CA_s, T-T_s]).
        """
        self.acados_ocp_solver.set(0, "lbx", x0_full)
        self.acados_ocp_solver.set(0, "ubx", x0_full)

    def solve(self, x0_full: np.ndarray, ref: Optional[np.ndarray] = None):
        """
        Set initial state, optionally update the reference, solve the OCP,
        and return the first control input.
        """
        self.set_initial_state(x0_full)
        if ref is not None:
            self._set_reference(ref)
        status = self.acados_ocp_solver.solve()
        if status != 0:
            print(f"[ReactorMPC] Solver returned status {status}.")
        return self.acados_ocp_solver.get(0, "u")

    def get_planned_trajectory(self):
        """
        Retrieve the planned state and input trajectories.
        For multi-phase, note there is an extra transition stage.
        """
        N = self.opts.N
        if self.multi_phase:
            N_total = N + 1
        else:
            N_total = N
        traj_x, traj_u = [], []
        for i in range(N_total):
            traj_x.append(self.acados_ocp_solver.get(i, "x"))
            traj_u.append(self.acados_ocp_solver.get(i, "u"))
        traj_x.append(self.acados_ocp_solver.get(N_total, "x"))
        return traj_x, traj_u

    ###########################################################################
    # Model Building Routines
    ###########################################################################

    def _build_full_model(self) -> AcadosModel:
        """
        Build the full model for Phase 0.
        State: x = [CA_rel, T_rel]  (2D, where CA_rel = CA - CA_s, T_rel = T - T_s)
        Input: u = [CA0, Q] (absolute values)
        Dynamics: defined based on the reactor equations written relative to steady state.
        """
        model = AcadosModel()
        model.name = f"chem_full_{self.timestamp}"
        x = ca.MX.sym("x", 2)   # [CA_rel, T_rel]
        u = ca.MX.sym("u", 2)   # [CA0, Q]
        # Unpack state and inputs
        CA_rel = x[0]
        T_rel  = x[1]
        CA = self.opts.CA_s + CA_rel
        T  = self.opts.T_s + T_rel
        CA0 = u[0]
        Q   = u[1]
        # For simplicity, define the dynamics (in relative form):
        # d(CA_rel)/dt = F/V*(CA0 - CA) - k0*exp(-E/(R_const*T))*(CA)**2
        # d(T_rel)/dt  = F/V*(T0 - T) + alpha*k0*exp(-E/(R_const*T))*(CA)**2 + Q/(rho*cp*V)
        F = self.opts.F
        V = self.opts.V
        T0 = self.opts.T0
        k0 = self.opts.k0
        E = self.opts.E
        R_const = self.opts.R_const
        # For temperature dynamics, include a constant factor alpha to account for reaction enthalpy,
        # here simplified as a constant.
        alpha = -self.opts.delta_H / (self.opts.rho * self.opts.cp * V)

        CA_dot = F/V*(CA0 - CA) - k0 * ca.exp(-E/(R_const*T)) * CA**2
        T_dot  = F/V*(T0 - T) + alpha * k0 * ca.exp(-E/(R_const*T)) * CA**2 + Q/(self.opts.rho*self.opts.cp*V)

        xdot = ca.vertcat(CA_dot, T_dot)
        model.x = x
        model.u = u
        model.xdot = ca.MX.sym("xdot", 2)
        model.f_expl_expr = xdot
        model.f_impl_expr = model.xdot - model.f_expl_expr
        return model

    def _build_transition_model(self) -> AcadosModel:
        """
        Build a discrete transition model for Phase 1.
        Maps full state [CA_rel, T_rel] to a reduced state [CA_rel].
        """
        x = ca.SX.sym("x", 2)  # full state: [CA_rel, T_rel]
        # The discrete mapping: output = CA_rel.
        disc_expr = x[0]
        model = AcadosModel()
        model.name = f"chem_trans_{self.timestamp}"
        model.x = x
        # In discrete dynamics, we define disc_dyn_expr instead of f_expl_expr.
        model.disc_dyn_expr = disc_expr
        return model

    def _build_reduced_model(self) -> AcadosModel:
        """
        Build the reduced model for Phase 2.
        Here, the fast dynamics are neglected, leading to an algebraic equation.
        State: x = [CA_rel] (1D)
        Input: u = [CA0] (only reactant feed, since heating is set to zero)
        The dynamics are defined for the slow evolution of CA relative.
        """
        model = AcadosModel()
        model.name = f"chem_reduced_{self.timestamp}"
        x = ca.MX.sym("x", 1)  # only CA_rel
        u = ca.MX.sym("u", 1)  # reactant feed input
        CA_rel = x[0]
        CA = self.opts.CA_s + CA_rel
        # In Phase 2, T is assumed to be fixed at the slow manifold.
        F = self.opts.F
        V = self.opts.V
        k0 = self.opts.k0
        E = self.opts.E
        R_const = self.opts.R_const
        # Dynamics: d(CA_rel)/dt = F/V*(u - CA) - k0*exp(-E/(R_const*T))*(CA)**2.
        CA_dot = F/V*(u - CA) - k0 * ca.exp(-E/(R_const*self.algebraic_expr_polynomial(x))) * CA**2
        model.x = x
        model.u = u
        model.xdot = ca.MX.sym("xdot", 1)
        model.f_expl_expr = ca.vertcat(CA_dot)
        model.f_impl_expr = model.xdot - model.f_expl_expr
        return model

    ###########################################################################
    # Cost Expressions
    ###########################################################################

    def _cost_expr_full(self, model: AcadosModel) -> ca.MX:
        """
        Define the stage cost for Phase 0 (full model).
        Cost vector: [L_econ, u0, u1] where
          L_econ = - k0*exp(-E/(R_const*(T_s+T_rel)))*(CA_s + CA_rel)**2.
        """
        x_sym = model.x
        u_sym = model.u
        CA_rel = x_sym[0]
        T_rel  = x_sym[1]
        CA = self.opts.CA_s + CA_rel
        T = self.opts.T_s + T_rel
        k0 = self.opts.k0
        E = self.opts.E
        R_const = self.opts.R_const

        L_econ = k0 * ca.exp(-E/(R_const*T)) * CA**2  # production rate
        cost_expr = ca.vertcat(L_econ, u_sym[0], u_sym[1])
        return cost_expr

    def _cost_expr_transition(self, model: AcadosModel) -> ca.MX:
        """
        Cost for Phase 1 (transition).
        Here, we penalize the deviation in temperature: cost = T_rel.
        (This will be squared by the least-squares formulation.)
        """
        x_sym = model.x  # x = [CA_rel, T_rel]
        T_rel  = x_sym[1]
        T = self.opts.T_s + T_rel
        cost_expr = T - self.algebraic_expr_polynomial(x_sym)  # T_rel
        return cost_expr

    def _cost_expr_reduced(self, model: AcadosModel) -> ca.MX:
        """
        Cost for Phase 2 (reduced model).
        With T fixed to T_s, the economic cost becomes:
          L_econ_reduced = - k0*exp(-E/(R_const*T_s))*(CA_s + CA_rel)**2.
        The cost vector is [L_econ_reduced, u] (only reactant feed input).
        """
        x_sym = model.x  # 1D: [CA_rel]
        u_sym = model.u  # 1D: reactant feed
        CA_rel = x_sym[0]
        CA = self.opts.CA_s + CA_rel
        k0 = self.opts.k0
        E = self.opts.E
        R_const = self.opts.R_const

        L_econ_red = k0 * ca.exp(-E/(R_const*self.algebraic_expr_polynomial(x_sym))) * CA**2 # production rate
        cost_expr = ca.vertcat(L_econ_red, u_sym[0])
        return cost_expr

    ###########################################################################
    # OCP Formulations
    ###########################################################################

    def _create_single_phase_ocp_full(self) -> AcadosOcp:
        """
        Single-phase OCP using the full model for the entire horizon.
        """
        ocp = AcadosOcp()
        model = self._build_full_model()
        ocp.model = model
        ocp.solver_options.N_horizon = self.opts.N
        ocp.solver_options.tf = sum(self.opts.step_sizes)

        cost_expr = self._cost_expr_full(model)
        ocp.model.cost_y_expr = cost_expr
        ocp.cost.cost_type = "NONLINEAR_LS"
        # Terminal cost on [L_econ, u0, u1] – for simplicity, use same cost as stage
        ocp.model.cost_y_expr_e = cost_expr[0]  # or select subset if desired
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        # Weighting: build a block-diagonal matrix: weight on L_econ, then on u.
        W_stage = np.diag([self.opts.Q_econ, self.opts.R_reactant, self.opts.R_heating])
        ocp.cost.W = W_stage
        ocp.cost.W_e = np.diag([self.opts.Q_econ])
        ocp.cost.yref = np.zeros(3)
        ocp.cost.yref_e = np.zeros(1)

        # State constraints on x = [CA_rel, T_rel]
        ocp.constraints.x0 = np.zeros(2)
        ocp.constraints.idxbx = np.array([0, 1])
        ocp.constraints.lbx = np.array([self.opts.CA_rel_min, self.opts.T_rel_min])
        ocp.constraints.ubx = np.array([self.opts.CA_rel_max, self.opts.T_rel_max])

        # Input constraints: u = [CA0, Q]
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.lbu = np.array([self.opts.CA0_min, self.opts.Q_min])
        ocp.constraints.ubu = np.array([self.opts.CA0_max, self.opts.Q_max])

        self._set_solver_options(ocp)
        return ocp

    def _create_multi_phase_ocp(self) -> AcadosMultiphaseOcp:
        """
        Multi-phase OCP with three phases:
         - Phase 0: Full model (Phase 0) for 'switch_stage' intervals.
         - Phase 1: Transition (1 interval): map [CA_rel, T_rel] to CA_rel and penalize T_rel.
         - Phase 2: Reduced model for the remaining intervals.
        """
        N0 = self.opts.switch_stage       # Phase 0 horizon
        N2 = self.opts.N - N0             # Phase 2 horizon
        mp_ocp = AcadosMultiphaseOcp([N0, 1, N2])

        # ----- Phase 0: Full model -----
        ocp0 = AcadosOcp()
        model0 = self._build_full_model()
        ocp0.model = model0
        cost_expr0 = self._cost_expr_full(model0)
        ocp0.model.cost_y_expr = cost_expr0
        ocp0.cost.cost_type = "NONLINEAR_LS"
        # For Phase 0, we use the same stage cost and weighting as in the single-phase full model
        W0 = np.diag([self.opts.Q_econ, self.opts.R_reactant, self.opts.R_heating])
        ocp0.cost.W = W0
        ocp0.cost.yref = np.zeros(3)
        # Set initial state constraints
        nx0 = model0.x.size()[0]
        ocp0.constraints.x0 = np.zeros(nx0)
        ocp0.constraints.idxbx = np.array([0, 1])
        ocp0.constraints.lbx = np.array([self.opts.CA_rel_min, self.opts.T_rel_min])
        ocp0.constraints.ubx = np.array([self.opts.CA_rel_max, self.opts.T_rel_max])
        # Input constraints for Phase 0
        ocp0.constraints.idxbu = np.array([0, 1])
        ocp0.constraints.lbu = np.array([self.opts.CA0_min, self.opts.Q_min])
        ocp0.constraints.ubu = np.array([self.opts.CA0_max, self.opts.Q_max])
        self._set_solver_options(ocp0)

        # ----- Phase 1: Transition -----
        ocp1 = AcadosOcp()
        model1 = self._build_transition_model()
        ocp1.model = model1
        cost_expr1 = self._cost_expr_transition(model1)  # scalar = T_rel
        ocp1.model.cost_y_expr = cost_expr1
        ocp1.cost.cost_type = "NONLINEAR_LS"
        ocp1.cost.W = np.array([[self.opts.Q_trans]])
        ocp1.cost.yref = np.array([0.0])
        self._set_solver_options(ocp1)

        # ----- Phase 2: Reduced Model -----
        ocp2 = AcadosOcp()
        model2 = self._build_reduced_model()
        ocp2.model = model2
        cost_expr2 = self._cost_expr_reduced(model2)
        ocp2.model.cost_y_expr = cost_expr2
        ocp2.cost.cost_type = "NONLINEAR_LS"
        # Terminal cost: we take only the economic term
        ocp2.model.cost_y_expr_e = cost_expr2[0]
        ocp2.cost.cost_type_e = "NONLINEAR_LS"
        W2 = np.diag([self.opts.Q_econ, self.opts.R_reactant])
        ocp2.cost.W = W2
        ocp2.cost.yref = np.zeros(2)
        ocp2.cost.W_e = np.diag([self.opts.Q_econ])
        ocp2.cost.yref_e = np.zeros(1)
        # State constraint on reduced state: x = [CA_rel]
        ocp2.constraints.idxbx = np.array([0])
        ocp2.constraints.lbx = np.array([self.opts.CA_rel_min])
        ocp2.constraints.ubx = np.array([self.opts.CA_rel_max])
        # Input constraint: u (reactant feed)
        ocp2.constraints.idxbu = np.array([0])
        ocp2.constraints.lbu = np.array([self.opts.CA0_min])
        ocp2.constraints.ubu = np.array([self.opts.CA0_max])
        self._set_solver_options(ocp2)

        # Combine phases
        mp_ocp.set_phase(ocp0, 0)
        mp_ocp.set_phase(ocp1, 1)
        mp_ocp.set_phase(ocp2, 2)
        self._set_solver_options(mp_ocp)

        # Set overall time steps: include one extra step for the transition phase
        step_sizes_full = list(self.opts.step_sizes[:N0]) + [1.0] + list(self.opts.step_sizes[N0:])
        mp_ocp.solver_options.time_steps = np.array(step_sizes_full)
        mp_ocp.solver_options.tf = sum(step_sizes_full)
        # For multi-phase, you can also set phase-specific integrator types.
        mp_ocp.mocp_opts.integrator_type = [
            self.opts.integrator_type,   # Phase 0
            "DISCRETE",                 # Phase 1
            "ERK",                      # Phase 2 (assumed to be fast enough)
        ]
        return mp_ocp

    ###########################################################################
    # Solver Options
    ###########################################################################

    def _set_solver_options(self, ocp_obj):
        """
        Set common solver options.
        """
        ocp_obj.solver_options.nlp_solver_type = self.opts.nlp_solver_type
        ocp_obj.solver_options.qp_solver = self.opts.qp_solver
        ocp_obj.solver_options.integrator_type = self.opts.integrator_type
        ocp_obj.solver_options.levenberg_marquardt = self.opts.levenberg_marquardt
        ocp_obj.solver_options.tol = 1e-6
        ocp_obj.solver_options.qp_tol = 1e-2 * ocp_obj.solver_options.tol
        ocp_obj.solver_options.print_level = 1
        # Other options (like hpipm_mode and globalization) can be set as needed:
        ocp_obj.solver_options.hpipm_mode = "ROBUST"
        ocp_obj.solver_options.globalization = "MERIT_BACKTRACKING"

    ###########################################################################
    # Reference Setting
    ###########################################################################

    def _set_reference(self, target_production_rate: np.ndarray):
        """
        """
        N = self.opts.N
        offset = 0
        if not self.multi_phase:
            # For single-phase full model, set reference for each stage.
            for stage in range(N):
                self.acados_ocp_solver.set(stage, "yref", np.array([target_production_rate, 0.0, 0.0]))
            self.acados_ocp_solver.set(N, "yref", np.array([target_production_rate]))
        else:
            # For multi-phase:
            # Phase 0: for stages 0,..., N0-1
            N0 = self.opts.switch_stage
            for stage in range(N0):
                self.acados_ocp_solver.set(stage, "yref", np.array([target_production_rate, 0.0, 0.0]))
            # Phase 1: transition phase (index N0)
            self.acados_ocp_solver.set(N0, "yref", np.array([0.0]))
            # Phase 2: for remaining stages
            for stage in range(N0+1, N+1):
                self.acados_ocp_solver.set(stage, "yref", np.array([target_production_rate, 0.0]))
            self.acados_ocp_solver.set(N+1, "yref", np.array([target_production_rate]))
    
    ###########################################################################
    # Simulation Solver (Optional)
    ###########################################################################

    def _create_sim_full_model(self):
        """
        Create an AcadosSimSolver for the full model (Phase 0), useful for closed-loop simulation.
        """
        sim = AcadosSim()
        model = self._build_full_model()
        sim.model = model
        sim.solver_options.T = self.opts.step_sizes[0]
        sim.solver_options.integrator_type = self.opts.integrator_type
        json_file = f"acados_sim_full_{self.timestamp}.json"
        self.sim_solver = AcadosSimSolver(sim, json_file=json_file)

    def _fit_algebraic_polynomial(self):
        """
        Fit a poly_order-th order polynomial T = T(x) to the ground truth relationship that defines
        the slow manifold in the reduced model. In the reduced model, we replace T by T(x[0]),
        where x[0] is the relative concentration (CA - CA_s). The ground truth is obtained by solving
        the algebraic equation (with Q = 0):
        
            F/V*(T0 - T) + α * k0 * exp(-E/(R_const*T)) * (CA_s + x)^2 + Q_s/(rho*cp*V) = 0
        
        for T, with x = CA_rel. The function then fits a polynomial of order poly_order over a range
        [-poly_fit_range, poly_fit_range] and stores the result as a CasADi function.
        Additionally, it plots the data points alongside the fitted polynomial.
        """
        # Unpack necessary parameters from options:
        poly_order = self.opts.poly_order
        poly_range = self.opts.poly_fit_range
        num_samples = 20
        x_samples = np.linspace(-poly_range, poly_range, num_samples)
        T_samples = []
        
        F = self.opts.F
        V = self.opts.V
        T0 = self.opts.T0
        k0 = self.opts.k0
        E = self.opts.E
        R_const = self.opts.R_const
        Q_s = self.opts.Q_s  # assumed to be provided in opts
        CA_s = self.opts.CA_s
        # Define alpha as in your full model (assuming delta_H, rho, cp, and V are provided)
        alpha = -self.opts.delta_H / (self.opts.rho * self.opts.cp * V)
        
        # Define the algebraic equation f(T) = 0 for a given CA_rel:
        def T_equation(T, CA_rel):
            CA = CA_s + CA_rel
            return F/V*(T0 - T) + alpha * k0 * np.exp(-E/(R_const*T)) * CA**2 + Q_s/(self.opts.rho*self.opts.cp*V)
        
        # For each sample point, solve for T (initial guess: T0)
        for x_val in x_samples:
            T_sol = fsolve(lambda T: T_equation(T, x_val), T0)[0]
            T_samples.append(T_sol)
        
        # Fit a polynomial of order poly_order to the data
        coeffs = np.polyfit(x_samples, T_samples, poly_order)
        # Construct a CasADi symbolic polynomial p(x) = a0*x^k + a1*x^(k-1) + ... + a_k.
        x_sym = ca.MX.sym("x", 1)
        poly_expr = 0
        for i, a_i in enumerate(coeffs):
            power = poly_order - i
            poly_expr += a_i * x_sym**power
        
        # Save the polynomial as a CasADi function:
        self.algebraic_expr_polynomial = ca.Function("algebraic_expr_polynomial", [x_sym], [poly_expr])

        # Optionally, store an interpolant for the ground truth:
        # self.ground_truth_T_function = ca.interpolant("ground_truth_T", "linear", [x_samples], T_samples)
        
        # --- Plot the data points and fitted polynomial ---
        # Create a fine grid for plotting the polynomial:
        x_fine = np.linspace(-poly_range, poly_range, 200)
        # Evaluate the polynomial on the fine grid:
        poly_eval = np.array(self.algebraic_expr_polynomial(x_fine).full().flatten())
        
        plt.figure(figsize=(6,4))
        plt.plot(x_fine, poly_eval, label="Fitted Polynomial", linewidth=2)
        plt.plot(x_samples, T_samples, "ro", label="Data Points")
        plt.xlabel("Relative Concentration (CA - CA_s)")
        plt.ylabel("Approximated Temperature T")
        plt.title("Polynomial Fit of T(x) on the Slow Manifold")
        plt.legend()
        plt.grid(True)
        plt.show()


