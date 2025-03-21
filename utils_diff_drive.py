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
    solve_times = []
    
    # Initialize Ornstein-Uhlenbeck noise
    noise_state = np.zeros(nx)
    
    for step in range(num_steps):
        
        if step % control_step == 0:
            u_opt = mpc.solve(x_traj[step, :nx0_mpc])
            solve_times.append(mpc.acados_ocp_solver.get_stats('time_tot')) 
        
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
    
    return x_traj, u_traj, stage_costs, solve_times

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect

def compute_exponential_step_sizes(dt_initial, T_total, N_steps, plot=False):
    """
    Compute exponentially increasing step sizes such that their sum equals T_total.

    Args:
        dt_initial (float): Initial step size (e.g. 0.002).
        T_total (float): Total time horizon (e.g. 20.0).
        N_steps (int): Number of steps (e.g. 100).
        plot (bool): Whether to plot the step sizes (default: False).

    Returns:
        np.ndarray: Array of step sizes of length N_steps.
    """

    # Function to find r: the common ratio of the geometric series
    def geometric_sum_error(r):
        if np.isclose(r, 1.0):
            return dt_initial * N_steps - T_total
        return dt_initial * (1 - r**N_steps) / (1 - r) - T_total

    # Find r using root-finding
    r = bisect(geometric_sum_error, 0.9, 5.0)

    # Generate step sizes
    step_sizes = np.array([dt_initial * r**i for i in range(N_steps)])

    # Optional plot
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(range(N_steps), step_sizes, marker='o')
        plt.xlabel("Step Index")
        plt.ylabel("Step Size")
        plt.title("Exponential Step Size Growth")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return step_sizes
