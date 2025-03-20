import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def plot_diff_drive_trajectory(y_ref, closed_loop_traj=None, open_loop_plan=None):
    """
    Plots the 2D trajectory of the differential drive robot.
    
    Args:
        y_ref (np.ndarray): Target state [x_ref, y_ref].
        closed_loop_traj (np.ndarray, optional): Closed-loop trajectory of shape (time_steps, 2).
        open_loop_plan (np.ndarray, optional): Open-loop MPC plan of shape (time_steps, 2).
    """
    plt.figure(figsize=(8, 6))
    
    # Plot the target state as a reference point
    plt.scatter(y_ref[0], y_ref[1], color='gold', marker='*', s=200, label="Target State $y_{ref}$")
    
    # Plot closed-loop trajectory if provided
    if closed_loop_traj is not None:
        x_closed, y_closed = closed_loop_traj[:, 0], closed_loop_traj[:, 1]
        plt.plot(x_closed, y_closed, 'r-', label="Closed Loop Trajectory", linewidth=2)
        
        # Add arrows for orientation
        for i in range(0, len(x_closed), max(1, len(x_closed) // 10)):
            theta = closed_loop_traj[i, 3]  # Assuming theta is stored at index 3
            dx, dy = np.cos(theta) * 0.1, np.sin(theta) * 0.1
            plt.arrow(x_closed[i], y_closed[i], dx, dy, head_width=0.05, color='r')
    
    # Plot open-loop trajectory if provided
    if open_loop_plan is not None:
        x_open, y_open = open_loop_plan[:, 0], open_loop_plan[:, 1]
        plt.plot(x_open, y_open, 'b--', label="Open Loop Plan", linewidth=2)
        
        # Add arrows for orientation
        for i in range(0, len(x_open), max(1, len(x_open) // 10)):
            theta = open_loop_plan[i, 3]  # Assuming theta is stored at index 3
            dx, dy = np.cos(theta) * 0.1, np.sin(theta) * 0.1
            plt.arrow(x_open[i], y_open[i], dx, dy, head_width=0.05, color='b')
    
    # Labels and settings
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Differential Drive Robot Trajectory")
    plt.legend()
    plt.grid()
    plt.axis("equal")  # Ensures equal scaling for x and y axes
    plt.show()