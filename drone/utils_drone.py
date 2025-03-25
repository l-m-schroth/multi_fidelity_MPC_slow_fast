import numpy as np

def simulate_closed_loop_drone(
    x0, 
    mpc, 
    duration, 
    sigma_noise=0.0, 
    tau_noise=1.0, 
    sim_solver=None, 
    control_step=1
):
    """
    Simulate a closed-loop system for DroneMPC with zero-order hold (ZOH) control updates,
    supporting Ornstein-Uhlenbeck noise.

    Parameters
    ----------
    x0 : np.ndarray
        Initial state of dimension matching the sim solver (e.g. 12D for full model, 8D ignoring pendulum, etc.).
    mpc : DroneMPC
        The DroneMPC instance with .solve(...) method and .opts containing relevant Q, R, etc.
    duration : float
        Total simulation time in seconds.
    sigma_noise : float, optional
        Standard deviation of the OU process noise.
    tau_noise : float, optional
        Time constant controlling the OU noise correlation. Lower => higher frequency.
    sim_solver : AcadosSimSolver, optional
        If not provided, use e.g. mpc.acados_sim_solver from the DroneMPC class.
    control_step : int, optional
        Number of sim steps between MPC updates (ZOH in between).

    Returns
    -------
    x_traj : np.ndarray
        State trajectory of shape (num_steps+1, x_dim).
    u_traj : np.ndarray
        Applied controls of shape (num_steps, u_dim).
    stage_costs : list of float
        List of scalar stage costs at each step.
    solve_times : list of float
        Time taken by each MPC solve (for profiling).
    """

    # If no external sim_solver is provided, use the drone's default
    if sim_solver is None:
        # For example, if your DroneMPC created a sim solver for the full model:
        sim_solver = mpc.sim_solver_full if hasattr(mpc, 'sim_solver_full') else None
        if sim_solver is None:
            raise ValueError("No sim_solver provided, and mpc.sim_solver_full not found.")

    dt_sim = sim_solver.T  # simulation step size from the solver
    num_steps = int(duration / dt_sim)

    x_dim = 12
    u_dim = 2

    # Allocate storage for the trajectory
    x_traj = np.zeros((num_steps + 1, x_dim))
    x_traj[0] = x0
    u_traj = np.zeros((num_steps, u_dim))

    stage_costs = []
    solve_times = []

    # Ornstein-Uhlenbeck noise state
    noise_state = np.zeros(x_dim)

    # We'll keep a "current" control guess
    u_current = np.zeros(u_dim)

    # Main loop
    for step in range(num_steps):
        # MPC update every `control_step` steps
        if step % control_step == 0:
            # Solve the MPC with the current state
            u_current = mpc.solve(x_traj[step])
            # For timing stats (if supported)
            solve_time = mpc.acados_ocp_solver.get_stats('time_tot')
            solve_times.append(solve_time)
     
        # Apply the control
        u_traj[step] = u_current

        # Compute the stage cost
        cost_val = 0 #get_drone_stage_cost(x_traj[step], u_current, mpc)
        stage_costs.append(cost_val)

        # Simulate one step
        sim_solver.set("x", x_traj[step])
        sim_solver.set("u", u_current)
        status = sim_solver.solve()
        if status != 0:
            print(f"Warning: sim solver returned status {status} at step {step}")

        # Update OU noise
        # dx_noise = -noise_state*(dt_sim/tau_noise) + sigma_noise*...
        dt = dt_sim
        noise_state += (-noise_state * dt / tau_noise) \
                       + (sigma_noise * np.sqrt(2*dt/tau_noise) * np.random.randn(x_dim))

        # Next state
        x_next = sim_solver.get("x") + noise_state
        x_traj[step+1] = x_next

    return x_traj, u_traj, stage_costs, solve_times


def get_drone_stage_cost(x, u, mpc):
    x_dim = x.shape[0]
    u_dim = u.shape[0]

    # We retrieve some options from mpc.opts
    # Adjust the names if your DroneMPC uses different ones
    if hasattr(mpc.opts, 'Q_mat_full'):
        Q_full = mpc.opts.Q_mat_full
    else:
        # fallback
        Q_full = np.eye(x_dim)

    if hasattr(mpc.opts, 'Q_acc'):
        # e.g. ignoring pend => we might do x^T Q_approx x
        Q_approx = np.eye(x_dim)  # as fallback
    else:
        Q_approx = np.eye(x_dim)

    # A small R for the input
    if hasattr(mpc.opts, 'R_reg'):
        R_reg = mpc.opts.R_reg
        if R_reg.shape[0] != u_dim:
            R_reg = np.eye(u_dim) * 1e-3
    else:
        R_reg = np.eye(u_dim) * 1e-3

    # Decide which cost:
    if x_dim >= 12:
        # full pendulum model => x.T Q_full x + u.T R_reg u
        # user might have or might want extra terms
        cost = float(x @ Q_full @ x + u @ R_reg @ u)
    elif x_dim >= 8:
        # ignoring pend => let's do x^T Q_approx x + u^T R_reg u
        # you might have an approximate Q for 8D states
        # We'll just do something
        Q_approx_resized = Q_approx
        if Q_approx_resized.shape[0] != x_dim:
            # fallback or slice
            Q_approx_resized = np.eye(x_dim)
        cost = float(x @ Q_approx_resized @ x + u @ R_reg @ u)
    else:
        # direct thrust => dimension <8
        # e.g. 6D => do x^T Q + u^T R
        cost = float(x @ np.eye(x_dim) @ x + u @ R_reg @ u)

    return cost
