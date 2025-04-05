import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from acados_template import latexify_plot

def plot_diff_drive_trajectory(y_ref, mpc=None, closed_loop_traj=None, open_loop_plan=None, plot_errors=False, latexify=False):
    """
    Plots the 2D trajectory of the differential drive robot.
    
    Args:
        y_ref (np.ndarray): Target state [x_ref, y_ref].
        closed_loop_traj (np.ndarray, optional): Closed-loop trajectory of shape (time_steps, 2).
        open_loop_plan (np.ndarray, optional): Open-loop MPC plan of shape (time_steps, 2).
        plot_errors (bool, optional): If True, plots orientation arrows along the trajectory. Default is False.
    """
    if latexify:
        latexify_plot()
    plt.figure(figsize=(8, 6))
    
    # Plot the target state as a reference point
    plt.scatter(y_ref[0], y_ref[1], color='gold', marker='*', s=200, label="Target State $y_{ref}$")
    
    # Plot closed-loop trajectory if provided
    if closed_loop_traj is not None:
        x_closed, y_closed = closed_loop_traj[:, 0], closed_loop_traj[:, 1]
        plt.plot(x_closed, y_closed, 'r-', label="Closed Loop Trajectory", linewidth=2)
        
        # Add arrows for orientation if plot_errors is True
        if plot_errors:
            for i in range(0, len(x_closed), max(1, len(x_closed) // 10)):
                theta = closed_loop_traj[i, 3]  # Assuming theta is stored at index 3
                dx, dy = np.cos(theta) * 0.1, np.sin(theta) * 0.1
                plt.arrow(x_closed[i], y_closed[i], dx, dy, head_width=0.05, color='r')
    
    # Plot open-loop trajectory if provided
    if open_loop_plan is not None:
        x_open = np.array([arr.squeeze()[0] for arr in open_loop_plan])
        y_open = np.array([arr.squeeze()[1] for arr in open_loop_plan])

        plt.plot(x_open, y_open, 'b--', label="Open Loop Plan", linewidth=2)
        
        # Add arrows for orientation if plot_errors is True
        if plot_errors:
            for i in range(0, len(x_open), max(1, len(x_open) // 10)):
                theta = open_loop_plan[i][3]  # Assuming theta is stored at index 3
                dx, dy = np.cos(theta) * 0.1, np.sin(theta) * 0.1
                plt.arrow(x_open[i], y_open[i], dx, dy, head_width=0.05, color='b')
    
    # Optionally visualize elliptical constraints
    if mpc is not None and mpc.opts.ellipse_centers is not None and mpc.opts.ellipse_half_axes is not None:
        for center, half_axes in zip(mpc.opts.ellipse_centers, mpc.opts.ellipse_half_axes):
            ellipse = patches.Ellipse(
                xy=center, width=2*half_axes[0], height=2*half_axes[1], 
                edgecolor='orange', facecolor='none', linestyle='dashed', linewidth=2
            )
            plt.gca().add_patch(ellipse)

    # Labels and settings
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Differential Drive Robot Trajectory")
    plt.legend()
    plt.grid()
    plt.axis("equal")  # Ensures equal scaling for x and y axes
    plt.show()

def plot_closed_loop_trajectories_and_inputs(
        baseline_x, baseline_u,
        mypic_x, mypic_u,
        our_x, our_u,
        dt,
        initial_state,
        goal_state,
        latexify=False):
    """
    Plots three closed-loop state trajectories and their corresponding control inputs.
    
    The trajectories are plotted in the x-y plane and the control inputs are shown in two subplots 
    (one for each input dimension) as a function of time.
    
    Args:
        baseline_x (np.ndarray): Baseline approach state trajectory (shape: [time_steps, state_dim]).
        baseline_u (np.ndarray): Baseline approach control inputs (shape: [time_steps, 2]).
        mypic_x (np.ndarray): Mypic MPC state trajectory (shape: [time_steps, state_dim]).
        mypic_u (np.ndarray): Mypic MPC control inputs (shape: [time_steps, 2]).
        our_x (np.ndarray): "Our Approach" state trajectory (shape: [time_steps, state_dim]).
        our_u (np.ndarray): "Our Approach" control inputs (shape: [time_steps, 2]).
        dt (float): Simulation time step (used to construct the time axis for control inputs).
        initial_state (array-like): Initial state as [x, y, heading]. Can be a flat list/array.
        goal_state (array-like): Goal state as [x, y, heading]. Can be a flat list/array.
        latexify (bool, optional): If True, applies LaTeX styling to the plots. Default is False.
    """
    # Optionally apply LaTeX styling (assuming latexify_plot is defined elsewhere)
    if latexify:
        latexify_plot(fontsize=12)
    
    # Ensure the initial and goal states are 1D arrays of shape (3,)
    initial_state = np.squeeze(np.array(initial_state))
    goal_state = np.squeeze(np.array(goal_state))
    
    # Compute the time vector from the control trajectory length
    n_steps = baseline_u.shape[0]
    time = np.arange(n_steps) * dt
    
    # Create a figure with three subplots:
    #  - The top subplot for the x-y trajectories.
    #  - Two subplots below for the control inputs (u[0] and u[1]) over time.
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # -----------------------------
    # Trajectory (x-y) plot
    # -----------------------------
    ax_traj = axs[0]
    
    # Plot baseline approach (solid black line)
    ax_traj.plot(baseline_x[:, 0], baseline_x[:, 1], 'k-', label='Baseline Approach', linewidth=2)
    
    # Plot Mypic MPC trajectory (solid blue line)
    ax_traj.plot(mypic_x[:, 0], mypic_x[:, 1], 'b-', label='Mypic MPC', linewidth=2)
    
    # Plot "Our Approach" (dashed black line so that it overlays the baseline)
    ax_traj.plot(our_x[:, 0], our_x[:, 1], 'k--', label='Our Approach', linewidth=2)
    
    # Plot the initial state as a grey dot with an arrow in the heading direction
    init_x, init_y, init_heading = initial_state[0], initial_state[1], initial_state[2]
    ax_traj.scatter(init_x, init_y, color='grey', s=100, zorder=5, label='Initial State')
    arrow_len = 0.2
    ax_traj.arrow(init_x, init_y, arrow_len * np.cos(init_heading), arrow_len * np.sin(init_heading),
                  head_width=0.05, head_length=0.1, fc='grey', ec='grey')
    
    # Plot the goal state as a golden star with an arrow in the heading direction
    goal_x, goal_y, goal_heading = goal_state[0], goal_state[1], goal_state[2]
    ax_traj.scatter(goal_x, goal_y, color='gold', marker='*', s=200, zorder=5, label='Goal State')
    ax_traj.arrow(goal_x, goal_y, arrow_len * np.cos(goal_heading), arrow_len * np.sin(goal_heading),
                  head_width=0.05, head_length=0.1, fc='gold', ec='gold')
    
    ax_traj.set_xlabel("X Position")
    ax_traj.set_ylabel("Y Position")
    ax_traj.set_title("Closed-Loop Trajectories")
    ax_traj.legend()
    ax_traj.grid(True)
    ax_traj.axis("equal")
    
    # -----------------------------
    # Control Input Plots
    # -----------------------------
    # u[0] plot
    ax_u0 = axs[1]
    ax_u0.plot(time, baseline_u[:, 0], 'k-', label='Baseline Approach', linewidth=2)
    ax_u0.plot(time, mypic_u[:, 0], 'b-', label='Mypic MPC', linewidth=2)
    ax_u0.plot(time, our_u[:, 0], 'k--', label='Our Approach', linewidth=2)
    ax_u0.set_ylabel("Control Input u[0]")
    ax_u0.set_title("Control Inputs Over Time")
    ax_u0.legend()
    ax_u0.grid(True)
    
    # u[1] plot
    ax_u1 = axs[2]
    ax_u1.plot(time, baseline_u[:, 1], 'k-', label='Baseline Approach', linewidth=2)
    ax_u1.plot(time, mypic_u[:, 1], 'b-', label='Mypic MPC', linewidth=2)
    ax_u1.plot(time, our_u[:, 1], 'k--', label='Our Approach', linewidth=2)
    ax_u1.set_xlabel("Time (s)")
    ax_u1.set_ylabel("Control Input u[1]")
    ax_u1.legend()
    ax_u1.grid(True)
    
    plt.tight_layout()
    plt.show()