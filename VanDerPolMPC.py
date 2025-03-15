from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver, AcadosMultiphaseOcp
import casadi as ca
import numpy as np

class VanDerPolMPC:
    """
    This class sets up a two-phase MPC for the modified Van der Pol oscillator.
    Phase 0: 3D model (x1, x2, x3).
    Phase 1: 2D model (x1, x2) only, neglecting x3.

    Also creates a (3D) AcadosSim solver for closed-loop simulation.
    """

    def __init__(
        self,
        # System parameters
        mu=1.0,
        epsilon_1=0.005,
        epsilon_2=0.3,
        epsilon_3=0.001,
        epsilon_4=0.01,
        d1=0.2,
        d2=0.1,
        d3=0.3,
        # MPC settings
        step_size_list=None,   # e.g. [0.02, 0.02, 0.02, ...]
        switch_stage=5,        # stage index at which we switch from 3D to 2D
        nlp_solver_type="SQP", # or "SQP_RTI"
        qp_solver="FULL_CONDENSING_HPIPM",
        # Cost weights
        Q_3d=np.diag([1.0, 1.0, 0.0]),  # e.g. penalize x1,x2 strongly, maybe x3 lightly or not at all
        R_3d=np.array([[0.01]]),       # single input u
        Q_2d=np.diag([1.0, 1.0]),
        R_2d=np.array([[0.01]])
    ):
        """
        Constructor for the Van der Pol MPC problem.
        
        :param mu, epsilon_i, d_i: Model parameters for the 3D system.
        :param step_size_list: List of time steps for each shooting interval (length = N).
        :param switch_stage: Stage index at which we switch from the 3D model to the 2D model.
        :param nlp_solver_type, qp_solver: Acados solver options.
        :param Q_3d, R_3d: Cost matrices for the 3D model stage cost.
        :param Q_2d, R_2d: Cost matrices for the 2D model stage cost.
        """
        # Store inputs
        if step_size_list is None:
            step_size_list = [0.02]*10  # default
        self.step_size_list = step_size_list
        self.N = len(step_size_list)   # horizon length
        self.switch_stage = switch_stage
        self.mu = mu
        self.eps1 = epsilon_1
        self.eps2 = epsilon_2
        self.eps3 = epsilon_3
        self.eps4 = epsilon_4
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        
        # Cost weighting
        self.Q_3d = Q_3d
        self.R_3d = R_3d
        self.Q_2d = Q_2d
        self.R_2d = R_2d
        
        # Create the multi-phase OCP (2 phases: 3D then 2D)
        # Each "phase" will have a separate model, cost dimension, etc.
        self._create_multiphase_ocp(nlp_solver_type, qp_solver)
        
        # Create the OCP solver
        json_file_name = f"acados_ocp_vdp_{self.N}_{switch_stage}.json"
        self.ocp_solver = AcadosOcpSolver(self.mpc_ocp, json_file=json_file_name)
        
        # Create an AcadosSim solver for the *3D model* for closed-loop simulation
        self.sim_solver_3d = self._create_sim_3d(json_file_suffix="vdp_3d_sim")

    def _create_multiphase_ocp(self, nlp_solver_type, qp_solver):
        """
        Builds a 2-phase AcadosMultiphaseOcp:
          Phase 0: from stage=0 to stage=(switch_stage-1) with 3D model
          Phase 1: from stage=switch_stage to stage=(N-1) with 2D model
        """
        # 1) Create the multi-phase container
        self.mpc_ocp = AcadosMultiphaseOcp(N_list=[self.switch_stage, self.N - self.switch_stage])
        total_dt = sum(self.step_size_list)
        self.mpc_ocp.solver_options.tf = total_dt  # total horizon length

        # 2) Define the two phases
        # Phase 0: 3D model
        ocp_phase_0 = self._build_ocp_3d(self.switch_stage)
        # Phase 1: 2D model
        ocp_phase_1 = self._build_ocp_2d(self.N - self.switch_stage)

        # 3) Insert the two phases into the multi-phase OCP
        self.mpc_ocp.set_phase(ocp_phase_0, 0)
        self.mpc_ocp.set_phase(ocp_phase_1, 1)

        # 4) Provide the time steps for each phase
        time_steps_phase_0 = np.array(self.step_size_list[:self.switch_stage])
        time_steps_phase_1 = np.array(self.step_size_list[self.switch_stage:])

        # The integrator type for each phase
        self.mpc_ocp.mocp_opts.integrator_type = ["IRK", "IRK"]

        # Assign time steps
        self.mpc_ocp.solver_options.time_steps = np.concatenate((time_steps_phase_0, time_steps_phase_1))

        # 5) Set overall solver options
        self.mpc_ocp.solver_options.nlp_solver_type = nlp_solver_type
        self.mpc_ocp.solver_options.qp_solver = qp_solver
        self.mpc_ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.mpc_ocp.solver_options.nlp_solver_max_iter = 200
        self.mpc_ocp.solver_options.print_level = 0
        self.mpc_ocp.solver_options.nlp_solver_tol_stat = 1e-6
        self.mpc_ocp.solver_options.nlp_solver_tol_eq = 1e-6
        self.mpc_ocp.solver_options.nlp_solver_tol_ineq = 1e-6
        self.mpc_ocp.solver_options.nlp_solver_tol_comp = 1e-6

    def _build_ocp_3d(self, N_phase):
        """
        Build an AcadosOcp object for the first phase (3D model),
        valid for N_phase shooting intervals.
        """
        ocp_3d = AcadosOcp()

        # 1) Create the 3D model
        model_3d = self._create_vdp_3d_model()
        ocp_3d.model = model_3d

        # 2) Dimensions
        nx_3d = 3   # (x1, x2, x3)
        nu_3d = 1   # input u
        ny_3d = nx_3d + nu_3d  # For nonlinear_ls cost: [x1, x2, x3, u]

        # 3) Cost
        ocp_3d.cost.cost_type = "NONLINEAR_LS"
        ocp_3d.cost.cost_type_e = "NONLINEAR_LS"
        
        # y = [x1, x2, x3, u]
        # We'll track x1, x2 vs reference, possibly x3 vs 0, and penalize control u
        x1, x2, x3 = model_3d.x[0], model_3d.x[1], model_3d.x[2]
        u_3d       = model_3d.u[0]

        ocp_3d.model.cost_y_expr = ca.vertcat(x1, x2, x3, u_3d)
        ocp_3d.model.cost_y_expr_e = ca.vertcat(x1, x2, x3)  # terminal stage does not include u

        # Weight matrix: W = block_diag(Q_3d, R_3d)
        # Q_3d is 3x3, R_3d is 1x1 => total 4x4
        W_3d = np.block([
            [self.Q_3d,             np.zeros((3,1))],
            [np.zeros((1,3)),       self.R_3d     ]
        ])

        ocp_3d.cost.W = W_3d
        ocp_3d.cost.W_e = self.Q_3d  # terminal cost on x only

        # Dummy references (updated at runtime)
        ocp_3d.cost.yref = np.zeros(ny_3d)
        ocp_3d.cost.yref_e = np.zeros(nx_3d)

        # 4)
