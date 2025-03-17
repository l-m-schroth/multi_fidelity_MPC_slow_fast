import numpy as np

def simulate_closed_loop(x0, mpc, duration, sigma_noise=0.0):
    # Compute number of steps based on the simulation solver step size
    dt_sim = mpc.opts.step_size_list[0]  
    num_steps = int(duration / dt_sim)

    # Initialize storage for state and input trajectories
    nx, nu = 3, 2
    x_traj = np.zeros((num_steps + 1, nx))
    u_traj = np.zeros((num_steps, nu))

    # Set initial state
    x_traj[0] = x0.squeeze()

    # is only 2d model is used, select only the x1 and x2 for the initial state of the MPC
    if mpc.opts.switch_stage == 0:
        nx0_mpc = 2
    else:
        nx0_mpc = 3 

    for step in range(num_steps):
        t = step * dt_sim

        # Solve MPC
        u_opt = mpc.solve(x_traj[step,:nx0_mpc], t)
        u_traj[step] = u_opt

        # Simulate one step using Acados Sim Solver
        mpc.acados_sim_solver_3d.set("x", x_traj[step])
        mpc.acados_sim_solver_3d.set("u", u_opt)
        mpc.acados_sim_solver_3d.solve()

        # Retrieve next state
        # to account for simulation time step, the noise variance is scaled by dt
        noise = np.random.normal(0, np.sqrt(sigma_noise*dt_sim), size=nx) 
        x_traj[step + 1] = mpc.acados_sim_solver_3d.get("x") + noise

    return x_traj, u_traj