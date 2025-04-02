import sys
import os
from acados_template import AcadosModel, AcadosOcp
import casadi as ca
from typing import Optional

# Add parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from drone_MPC import DroneMPC, DroneMPCOptions
import numpy as np

class ParametrizedDroneMPC(DroneMPC):
    """
    This class implements MPCs that smoothely interpolate between model switching and stage aggregation. This is used to compute the sensitivities with respect to the interpolation parameters via finite differences.
    We overwrite functions from the original class that need to change.
    NOTE: A LOT of copied code currently, just quick implementation. If I keep this, maybe restructure
    """        
    def _build_full_pendulum_model(self) -> AcadosModel:
        """
        Overwrites the original function. 
        """
        import eom_2d_quadro_springpend_explicit_casadi as eomfile_full
        import eom_2d_quadro_pend_explicit_casadi as eomfile_approx

        # full state x = [ y, z, phi, r, theta, y_dot, z_dot, phi_dot, r_dot, theta_dot, w1, w2 ]
        y = ca.MX.sym("y")
        z = ca.MX.sym("z")
        phi = ca.MX.sym("phi")
        r = ca.MX.sym("r")
        theta = ca.MX.sym("theta")
        y_dot = ca.MX.sym("y_dot")
        z_dot = ca.MX.sym("z_dot")
        phi_dot = ca.MX.sym("phi_dot")
        r_dot = ca.MX.sym("r_dot")
        theta_dot = ca.MX.sym("theta_dot")
        w1 = ca.MX.sym("w1")
        w2 = ca.MX.sym("w2")

        x_full = ca.vertcat(y, z, phi, r, theta, y_dot, z_dot, phi_dot, r_dot, theta_dot, w1, w2)
        x_reduced = ca.vertcat(y, z, phi, theta, y_dot, z_dot, phi_dot, theta_dot, w1, w2)

        u = ca.MX.sym("u", 2)
        p = ca.MX.sym("p", 1)  # interpolation parameter for model switching

        # Build the params vector for the ODE function (external forces = 0)
        params_list = [
            self.opts.M,          # 0 - M
            self.opts.m,          # 1 - m
            self.opts.Ixx,        # 2 - Ixx
            self.opts.g,          # 3 - g
            self.opts.k,          # 4 - spring stiffness (REMOVE FOR approx)
            self.opts.l0,         # 5 - rest length
            self.opts.c,          # 6 - rotor thrust constant
            self.opts.L_rot,      # 7 - half rotor distance
            0.0, 0.0, 0.0, 0.0    # 8–11 - external forces
        ]

        # Create full and approximate parameter vectors
        params_full   = ca.vertcat(*params_list)
        params_approx = ca.vertcat(*params_list[:4], *params_list[5:])  # omit index 4 (spring stiffness)

        # ODE from file
        x_dot_expr_full = eomfile_full.eom_2d_quadro_springpend_explicit(x_full, u, params_full)
        x_dot_expr_approx_10d = eomfile_approx.eom_2d_quadro_pend_explicit(x_reduced, u, params_approx)
        x_dot_expr_approx_padded = ca.vertcat(
            x_dot_expr_approx_10d[0:3],     # y_dot, z_dot, phi_dot
            ca.MX(0.0),                     # r_dot = 0
            x_dot_expr_approx_10d[3:7],     # theta_dot, y_ddot, z_ddot, phi_ddot
            ca.MX(0.0),                     # r_ddot = 0
            x_dot_expr_approx_10d[7:]       # theta_ddot, w1_dot, w2_dot
        )
        x_dot_expr = x_dot_expr_full*(1 - p) + x_dot_expr_approx_padded*p

        # Create the AcadosModel
        model = AcadosModel()
        model.name = f"drone_full_{self.timestamp}"
        model.x = x_full
        model.u = u
        model.p = p
        nx = x_full.shape[0]
        model.xdot = ca.MX.sym("xdot", nx)
        model.f_expl_expr = x_dot_expr
        model.f_impl_expr = model.xdot - model.f_expl_expr

        # ----------------------------------------------------------
        # Build a CasADi expression for the load acceleration:
        #   y_load = y + r*sin(theta)
        #   z_load = z - r*cos(theta)
        # then differentiate twice w.r.t. time
        y_drone = x_full[0]
        z_drone = x_full[1]
        r_val   = x_full[3]
        th_val  = x_full[4]

        y_load = y_drone + r_val*ca.sin(th_val)
        z_load = z_drone - r_val*ca.cos(th_val)

        # first derivative
        dy_load_dx = ca.jacobian(y_load, x_full)
        y_load_dot = dy_load_dx @ x_dot_expr
        dz_load_dx = ca.jacobian(z_load, x_full)
        z_load_dot = dz_load_dx @ x_dot_expr

        # second derivative
        dy_load_dot_dx = ca.jacobian(y_load_dot, x_full)
        y_load_ddot = dy_load_dot_dx @ x_dot_expr
        dz_load_dot_dx = ca.jacobian(z_load_dot, x_full)
        z_load_ddot = dz_load_dot_dx @ x_dot_expr

        # Store as a Function => model.load_acc_func( x_12 ) => [y_load_dd, z_load_dd]
        # We'll need this for the transition cost.
        self.load_acc_func = ca.Function(
            "load_acc_func",
            [model.x, model.u, model.p], [y_load_ddot, z_load_ddot],
            ["x_in", "u_in", "p_in"], ["y_load_dd", "z_load_dd"]
        )

        return model
    
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
        load_acc = self.load_acc_func(x_sym, model.u, model.p)  # => [y_load_dd, z_load_dd]
        y_load_ddot = load_acc[0]
        z_load_ddot = load_acc[1]

        u_1 = model.u[0]
        u_2 = model.u[1]

        cost_expr = ca.vertcat(
            y_drone, 
            z_drone, 
            ca.sign(y_load_ddot) * ca.fmin(1e12, ca.fabs(y_load_ddot)), 
            ca.sign(z_load_ddot) * ca.fmin(1e12, ca.fabs(z_load_ddot)), 
            u_1, 
            u_2
            )
        return cost_expr
    
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

        cost_expr_subs = ca.substitute(cost_expr, ocp.model.p, 0.0)
        diff = cost_expr_subs - yref_ca
        stage_cost_sym = diff.T @ W_stage @ diff

        # 4) Create a CasADi function
        self.stage_cost_func_full = ca.Function(
            "stage_cost_func_full",
            [ocp.model.x, ocp.model.u, yref_ca],
            [stage_cost_sym],
            ["x", "u", "y_ref"],
            ["stage_cost"]
        )

        # define parameter value
        ocp.parameter_values = np.array([[0.0]])  # default value for p

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
    
    def solve(self, x0: np.ndarray, p: np.ndarray, y_ref: Optional[np.ndarray] = None):
        """
        Overwrites solve from the original class
        """

        if self.opts.switch_stage == 0:
            x0 = self.to_10d(x0)

        # Phase 0 (index 0) gets x0 constraints
        self.acados_ocp_solver.set(0, "lbx", x0)
        self.acados_ocp_solver.set(0, "ubx", x0)

         # Set dynamic parameters stage-wise
        for stage in range(self.opts.N):
            self.acados_ocp_solver.set(stage, "p", np.array([p[stage]]))

        # If reference is provided, set it (assuming y_ref = [y*, z*])
        if y_ref is not None:
            self._set_reference_in_cost(y_ref)

        status = self.acados_ocp_solver.solve()
        if status != 0:
            print(f"[DroneMPC] solver returned status {status}.")

        # Return the first control input
        return self.acados_ocp_solver.get(0, "u")
    
    def _set_ocp_solver_options(self, ocp: AcadosOcp):
        ocp.solver_options.nlp_solver_type = self.opts.nlp_solver_type
        ocp.solver_options.qp_solver = self.opts.qp_solver
        ocp.solver_options.integrator_type = self.opts.integrator_type
        ocp.solver_options.nlp_solver_max_iter = 500
        ocp.solver_options.tol = 1e-7
        ocp.solver_options.qp_tol = 1e-2*ocp.solver_options.tol
        ocp.solver_options.print_level = 1
        ocp.solver_options.qp_solver_iter_max = 500
        ocp.solver_options.hessian_approx = "EXACT"#"GAUSS_NEWTON"
        ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        ocp.solver_options.levenberg_marquardt = self.opts.levenberg_marquardt # regularize Hessian stongly due to conditioning
        ocp.solver_options.hpipm_mode = "ROBUST" # problem seems a bit ill conditioned, try robust mode
        # ocp.solver_options.tf or time_steps must be set outside as appropriate.