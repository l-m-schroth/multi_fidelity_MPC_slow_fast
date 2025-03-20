import numpy as np

def simulate_closed_loop(x0, mpc, duration, sigma_noise=0.0, tau_noise=1.0, 
                         sim_solver=None, control_step=1):
    """
    Simulate a closed-loop system with zero-order hold (ZOH) control updates, supporting noise with adjustable frequency.
    
    Parameters:
        x0: np.array
            Initial state.
        mpc: object
            Model predictive controller with solve() method.
        duration: float
            Simulation duration in seconds.
        sigma_noise: float, optional
            Standard deviation of process noise.
        tau_noise: float, optional
            Time constant controlling the frequency of the noise. Lower values lead to higher-frequency noise.
        sim_solver: object, optional
            Acados simulation solver.
        control_step: int, optional
            Number of simulation steps between control updates.
    
    Returns:
        x_traj: np.array
            State trajectory over time.
        u_traj: np.array
            Control input trajectory over time.
        stage_costs: list
            List of stage costs over time.
    """
    
    if sim_solver is None:
        sim_solver = mpc.acados_sim_solver
        dt_sim = mpc.opts.step_sizes[0]  
    else:
        dt_sim = sim_solver.T
    num_steps = int(duration / dt_sim)
    
    nx, nu = 7, 2
    x_traj = np.zeros((num_steps + 1, nx))
    u_traj = np.zeros((num_steps, nu))
    x_traj[0] = x0.squeeze()
    nx0_mpc = nx
    u_opt = np.zeros(nu)
    stage_costs = []
    
    # Initialize Ornstein-Uhlenbeck noise
    noise_state = np.zeros(nx)
    
    for step in range(num_steps):
        
        if step % control_step == 0:
            u_opt = mpc.solve(x_traj[step, :nx0_mpc])
        
        u_traj[step] = u_opt

        I_r = x_traj[-1][-2]
        I_l = x_traj[-1][-1]
        V_r = u_traj[-1][0]
        V_l = u_traj[-1][1]
        stage_cost = x_traj[step].T @ mpc.opts.Q_mat_full @ x_traj[step] + np.abs(V_r * I_r) + np.abs(V_l * I_l)
        stage_costs.append(stage_cost)
        
        # Simulate one step using Acados Sim Solver
        sim_solver.set("x", x_traj[step])
        sim_solver.set("u", u_opt)
        sim_solver.solve()
        
        # Update Ornstein-Uhlenbeck noise
        noise_state += (-noise_state * dt_sim / tau_noise) + (sigma_noise * np.sqrt(2 * dt_sim / tau_noise) * np.random.randn(nx))
        
        # Retrieve next state with correlated noise
        x_traj[step + 1] = sim_solver.get("x") + noise_state
    
    return x_traj, u_traj, stage_costs