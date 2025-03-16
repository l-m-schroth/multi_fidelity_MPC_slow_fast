import numpy as np

def simulate_closed_loop(x0, mpc, duration):
    # Compute number of steps based on the simulation solver step size
    dt_sim = mpc.opts.step_size_list[0]  
    num_steps = int(duration / dt_sim)

    # Initialize storage for state and input trajectories
    nx, nu = 3, 2
    x_traj = np.zeros((num_steps + 1, nx))
    u_traj = np.zeros((num_steps, nu))

    # Set initial state
    x_traj[0] = x0.squeeze()

    for step in range(num_steps):
        t = step * dt_sim

        # Solve MPC
        u_opt = mpc.solve(x_traj[step], t)
        u_traj[step] = u_opt

        # Simulate one step using Acados Sim Solver
        mpc.acados_sim_solver_3d.set("x", x_traj[step])
        mpc.acados_sim_solver_3d.set("u", u_opt)
        mpc.acados_sim_solver_3d.solve()

        # Retrieve next state
        x_traj[step + 1] = mpc.acados_sim_solver_3d.get("x")

    return x_traj, u_traj