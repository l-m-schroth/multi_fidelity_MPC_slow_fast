import numpy as np

def simulate_closed_loop(x0, mpc, duration, sigma_noise=0.0, sim_solver=None, control_step=1):
    """
    Simulate a closed-loop system with zero-order hold (ZOH) control updates.
    
    Parameters:
        x0: np.array
            Initial state.
        mpc: object
            Model predictive controller with solve() method.
        duration: float
            Simulation duration in seconds.
        sigma_noise: float, optional
            Standard deviation of process noise.
        sim_solver: object, optional
            Acados simulation solver.
        control_step: int, optional
            Number of simulation steps between control updates.
            A value of 1 means MPC is updated at every simulation step.
    
    Returns:
        x_traj: np.array
            State trajectory over time.
        u_traj: np.array
            Control input trajectory over time.
    """
    
    # Acados Sim solver 
    if sim_solver is None:
        sim_solver = mpc.acados_sim_solver_3d
        # Compute number of steps based on the simulation solver step size
        dt_sim = mpc.opts.step_size_list[0]  
    else:
        dt_sim = sim_solver.T
    num_steps = int(duration / dt_sim)
    
    # Initialize storage for state and input trajectories
    nx, nu = 3, 2
    x_traj = np.zeros((num_steps + 1, nx))
    u_traj = np.zeros((num_steps, nu))
    
    # Set initial state
    x_traj[0] = x0.squeeze()

    # Determine MPC state size
    nx0_mpc = 2 if mpc.opts.switch_stage == 0 else 3 
    
    # Initialize control input (zero-order hold approach)
    u_opt = np.zeros(nu)
    stage_costs = []
    
    for step in range(num_steps):
        t = step * dt_sim
        
        # Solve MPC only every `control_step` steps
        if step % control_step == 0:
            u_opt = mpc.solve(x_traj[step, :nx0_mpc], t)
        
        u_traj[step] = u_opt

        # save stage costs
        stage_cost = x_traj[step][:2].T @ mpc.opts.Q_2d @ x_traj[step][:2] +  u_opt.T @ mpc.opts.R_2d @ u_opt
        stage_costs.append(stage_cost)

        # Simulate one step using Acados Sim Solver
        sim_solver.set("x", x_traj[step])
        sim_solver.set("u", u_opt)
        sim_solver.solve()
        
        # Retrieve next state with noise
        noise = np.random.normal(0, np.sqrt(sigma_noise * dt_sim), size=nx)
        x_traj[step + 1] = sim_solver.get("x") + noise
    
    return x_traj, u_traj, stage_costs