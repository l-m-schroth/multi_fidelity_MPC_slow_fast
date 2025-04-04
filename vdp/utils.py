import numpy as np
from plotting_utils import plot_phase_space

def simulate_closed_loop(x0, mpc, duration, sigma_noise=0.0, tau_noise=1.0, 
                         sim_solver=None, control_step=1, plot_open_loop_plan=False):
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
        sim_solver = mpc.acados_sim_solver_3d
        dt_sim = mpc.opts.step_size_list[0]  
    else:
        dt_sim = sim_solver.T
    num_steps = int(duration / dt_sim)
    
    nx, nu = 3, 2
    x_traj = np.zeros((num_steps + 1, nx))
    u_traj = np.zeros((num_steps, nu))
    x_traj[0] = x0.squeeze()
    nx0_mpc = 2 if mpc.opts.switch_stage == 0 else 3 
    u_opt = np.zeros(nu)
    stage_costs = []
    solve_times = []
    number_of_iters = []
    
    # Initialize Ornstein-Uhlenbeck noise
    noise_state = np.zeros(nx)
    
    for step in range(num_steps):
        t = step * dt_sim
        
        if step % control_step == 0:
            u_opt = mpc.solve(x_traj[step, :nx0_mpc], t)
            solve_times.append(mpc.acados_ocp_solver.get_stats('time_tot')) 
            number_of_iters.append(mpc.acados_ocp_solver.get_stats('sqp_iter'))
            if plot_open_loop_plan and step % 100*control_step == 0:
               plot_phase_space(mpc.Phi_t, mpc, open_loop_plan=True)
        
        u_traj[step] = u_opt
        
        x1_ref, x2_ref = mpc.Phi_t(t=t)
        x_ref = np.vstack((x1_ref, x2_ref)).squeeze()#
        stage_cost = (x_traj[step][:2] - x_ref).T @ mpc.opts.Q_2d @ (x_traj[step][:2] - x_ref) + u_opt.T @ mpc.opts.R_2d @ u_opt
        stage_costs.append(stage_cost)
        
        # Simulate one step using Acados Sim Solver
        sim_solver.set("x", x_traj[step])
        sim_solver.set("u", u_opt)
        sim_solver.solve()
        
        # Update Ornstein-Uhlenbeck noise
        noise_state += (-noise_state * dt_sim / tau_noise) + (sigma_noise * np.sqrt(2 * dt_sim / tau_noise) * np.random.randn(nx))
        
        # Retrieve next state with correlated noise
        x_traj[step + 1] = sim_solver.get("x") + noise_state
    
    return x_traj, u_traj, stage_costs, solve_times, number_of_iters