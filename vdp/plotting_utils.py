import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from acados_template import latexify_plot
import matplotlib.lines as mlines  # For custom legend handle
import os

def plot_phase_space(Phi_t, mpc=None, closed_loop_traj=None, open_loop_plan=False, latexify=False, legend=False, number=None, save=False):
    """
    Plots the phase space of the Van der Pol oscillator with the desired trajectory.
    Depending on the arguments, it can also overlay the open-loop plan (from MPC) or 
    the closed-loop trajectory.

    Args:
        Phi_t (function): Function generating the desired trajectory as (x1, x2) over time.
        mpc (VanDerPolMPC, optional): The MPC instance after calling solve().
                                      Required if open_loop_plan=True.
        closed_loop_traj (np.ndarray, optional): Closed-loop trajectory of shape (time_steps, 2).
                                                 Should contain only x1 and x2.
        open_loop_plan (bool, optional): If True, retrieves and plots the open-loop MPC plan.
        latexify (bool, optional): If True, applies latexified styles for plotting.
        legend (bool, optional): If True, shows the legend.
        number (int, optional): If given, places "number)" in the upper-left corner with 
                                a white bounding box for visibility.
    """

    # Define phase space grid
    X1, X2 = np.meshgrid(np.linspace(-2.5, 3, 20), np.linspace(-2.0, 2.5, 18))
    U = X2
    V = 1.0 * (1 - X1**2) * X2 - X1  # Van der Pol dynamics ignoring x3

    # Compute the desired trajectory Phi_t
    T = np.linspace(0, 50, 500)
    phi_x1, phi_x2 = np.array([Phi_t(t) for t in T]).T

    # Create the figure
    if latexify:
        latexify_plot()
    plt.figure(figsize=(8, 6))

    # Plot phase space vector field
    plt.streamplot(X1, X2, U, V, color="gray", arrowsize=1, density=0.8, linewidth=0.5)

    # Plot desired trajectory
    plt.plot(phi_x1, phi_x2, 'g', label="Desired Trajectory", linewidth=2)

    # Optionally overlay closed-loop trajectory
    if closed_loop_traj is not None:
        x1_closed, x2_closed = closed_loop_traj[:, 0], closed_loop_traj[:, 1]
        plt.plot(x1_closed, x2_closed, 'r', label="Closed Loop Trajectory", linewidth=2)

    # Optionally overlay open-loop plan
    if open_loop_plan and mpc is not None:
        x_plan, _ = mpc.get_planned_trajectory()
        switch_stage = mpc.opts.switch_stage

        if switch_stage == 0:  
            # **Only 2D Model Case**
            x1_plan = [x[0] for x in x_plan]
            x2_plan = [x[1] for x in x_plan]
            plt.plot(x1_plan, x2_plan, 'b--', label="Open Loop Plan (2D Model)", linewidth=2)

        elif switch_stage >= mpc.opts.N:  
            # **Only 3D Model Case**
            x1_plan = [x[0] for x in x_plan]
            x2_plan = [x[1] for x in x_plan]
            plt.plot(x1_plan, x2_plan, 'r--', label="Open Loop Plan (3D Model)", linewidth=2)

        else:  
            # **Mixed Model Case (3D → 2D switch)**
            x1_plan_before = [x[0] for x in x_plan[:switch_stage+1]]
            x2_plan_before = [x[1] for x in x_plan[:switch_stage+1]]
            x1_plan_after = [x[0] for x in x_plan[switch_stage+1:]]
            x2_plan_after = [x[1] for x in x_plan[switch_stage+1:]]
            plt.plot(x1_plan_before, x2_plan_before, 'r--', label="Open Loop Plan (3D Phase)", linewidth=2)
            plt.plot(x1_plan_after, x2_plan_after, 'b--', label="Open Loop Plan (2D Phase)", linewidth=2)

            # Add arrows indicating open-loop trajectory direction
            plt.quiver(
                x1_plan_before[::2], x2_plan_before[::2], 
                np.gradient(x1_plan_before)[::2], np.gradient(x2_plan_before)[::2], 
                angles="xy", scale_units="xy", scale=0.3, color="r", width=0.005, headwidth=4, headlength=6
            )
            plt.quiver(
                x1_plan_after[::2], x2_plan_after[::2], 
                np.gradient(x1_plan_after)[::2], np.gradient(x2_plan_after)[::2], 
                angles="xy", scale_units="xy", scale=0.3, color="b", width=0.005, headwidth=4, headlength=6
            )

    # Optionally visualize elliptical constraints (obstacle)
    if mpc is not None and mpc.opts.ellipse_centers is not None and mpc.opts.ellipse_half_axes is not None:
        for i, (center, half_axes) in enumerate(zip(mpc.opts.ellipse_centers, mpc.opts.ellipse_half_axes)):
            if i == 0:
                # Put the label on the first ellipse for the legend
                ellipse = patches.Ellipse(
                    xy=center,
                    width=2 * half_axes[0],
                    height=2 * half_axes[1],
                    facecolor='grey',
                    edgecolor='grey',
                    label='Obstacle'
                )
            else:
                ellipse = patches.Ellipse(
                    xy=center,
                    width=2 * half_axes[0],
                    height=2 * half_axes[1],
                    facecolor='grey',
                    edgecolor='grey',
                )
            plt.gca().add_patch(ellipse)

    # If 'number' is provided, place it in the upper-left corner with a white background
    if number is not None:
        plt.text(
            0.05, 0.95,           # Adjusted x to 0.05 to be on the left side
            f"{number})",
            transform=plt.gca().transAxes,
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=20,  # large font
            bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', pad=5)
        )

    # Labels and settings
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    #plt.title("Van der Pol Oscillator Phase Space")
    
    if legend:
        # Create a custom legend entry for the VDP phase space arrow.
        phase_arrow = mlines.Line2D([], [], color='gray', marker=r'$\rightarrow$', 
                                    linestyle='None', markersize=15, label='VDP phase space')
        # Get current legend handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()
        # Append the custom arrow legend entry
        handles.append(phase_arrow)
        labels.append('VDP phase space (phase diagram)')
        # Place the legend in the upper right corner
        plt.legend(handles=handles, loc='upper right')
    
    plt.grid()
    # Save figure if requested
    if save:
        os.makedirs("plots", exist_ok=True)
        filename = f"plots/vdp_trajectory_tracking_{number}.pgf"
        plt.savefig(filename, bbox_inches="tight")
    plt.show()





