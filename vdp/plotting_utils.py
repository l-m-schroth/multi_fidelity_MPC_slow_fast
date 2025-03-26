import numpy as np
import matplotlib.pyplot as plt

def plot_drone_mpc_solution(
    mpc, 
    reference_xy=None,
    closed_loop_traj=None,
    open_loop_plan=None,
    u_open_loop_plan=None,
    plot_title="Drone + Pendulum MPC",
    step_pose=10
):
    """
    Visualizes a 2D drone + pendulum MPC solution in three figures:

    Figure 1:
      - Big subplot: (y–z) trajectory
      - Smaller subplots: y(t), z(t), phi(t), r(t), theta(t)

    Figure 2:
      - Another (y–z) plot that draws the drone & pendulum lines at intervals
        to show the system configuration.

    Figure 3 (only if u_open_loop_plan is provided):
      - Top subplot: Left & Right thrust over time
      - Bottom subplot: Control inputs over time (dw1, dw2 or F1, F2)

    Arguments
    ---------
    mpc : DroneMPC
        The MPC instance (contains multi_phase, switch_stage, L_rot, etc.).
    reference_xy : (2,) array-like, optional
        The target [y*, z*].
    closed_loop_traj : np.ndarray, optional
        If open_loop_plan is None, we can pass a (T, dim) array of states.
    open_loop_plan : list of np.ndarray, optional
        Each element shape=(dim,). We unify them for plotting. If multi-phase,
        we color-code the segments. If single-phase, we treat it as one segment.
    u_open_loop_plan : list of np.ndarray, optional
        The control inputs for each stage (except maybe the terminal). Each element shape=(2,).
        Provide this if you want Figure 3 with thrust and input signals. Must align with open_loop_plan.
    plot_title : str, optional
        Title for the figure windows.
    step_pose : int, optional
        Interval for drawing the drone lines in Figure 2 to reduce clutter.
    """

    # ---------------------------
    # 1) Convert or unify the main open_loop_plan
    # ---------------------------
    if open_loop_plan is None and closed_loop_traj is not None:
        # convert (T,dim) -> list of (dim,)
        open_loop_plan = [row for row in closed_loop_traj]

    if open_loop_plan is None:
        print("No state trajectory data. Exiting.")
        return

    # A helper to unify a list of (dim,) snapshots -> (N, dim)
    def unify_snapshots_into_2d(snapshot_list):
        mats = []
        for s in snapshot_list:
            mats.append(s.reshape(1, -1))
        return np.vstack(mats)

    # We'll produce phase_data_list = [2D array for each phase], phase_colors = [...]
    if not mpc.multi_phase:
        # Single-phase => unify all into one 2D array
        phase_data_list = [unify_snapshots_into_2d(open_loop_plan)]
        phase_colors = ["blue"]
    else:
        # multi-phase => 3 phases
        N0 = mpc.opts.switch_stage
        snaps0 = open_loop_plan[:N0+1]
        snaps1 = [open_loop_plan[N0]] if (N0 < len(open_loop_plan)) else []
        snaps2 = open_loop_plan[N0+1:]
        p0_2d = unify_snapshots_into_2d(snaps0) if len(snaps0)>0 else None
        p1_2d = unify_snapshots_into_2d(snaps1) if len(snaps1)>0 else None
        p2_2d = unify_snapshots_into_2d(snaps2) if len(snaps2)>0 else None

        phase_data_list, phase_colors = [], []
        if p0_2d is not None and p0_2d.shape[0]>0:
            phase_data_list.append(p0_2d)
            phase_colors.append("red")
        if p1_2d is not None and p1_2d.shape[0]>0:
            phase_data_list.append(p1_2d)
            phase_colors.append("green")  # transition
        if p2_2d is not None and p2_2d.shape[0]>0:
            phase_data_list.append(p2_2d)
            phase_colors.append("blue")

    # ---------------------------------------------
    # FIGURE 1: Y–Z + states y,z,phi,r,theta
    # ---------------------------------------------
    fig1 = plt.figure(figsize=(10, 10))
    gs = fig1.add_gridspec(4, 2, height_ratios=[1.5,1,1,1])

    ax_xy = fig1.add_subplot(gs[0,:])
    ax_xy.set_title(f"{plot_title} - YZ Trajectory")
    ax_xy.set_xlabel("y [m]")
    ax_xy.set_ylabel("z [m]")
    ax_xy.grid(True)
    ax_xy.axis("equal")
    if reference_xy is not None:
        ax_xy.plot(reference_xy[0], reference_xy[1], marker='*', markersize=10, color='gold', label="Target")

    ax_y = fig1.add_subplot(gs[1,0])
    ax_y.set_title("y(t)")
    ax_z = fig1.add_subplot(gs[1,1])
    ax_z.set_title("z(t)")

    ax_phi = fig1.add_subplot(gs[2,0])
    ax_phi.set_title("phi(t)")
    ax_r   = fig1.add_subplot(gs[2,1])
    ax_r.set_title("r(t)")

    ax_th  = fig1.add_subplot(gs[3,0])
    ax_th.set_title("theta(t)")
    ax_unused = fig1.add_subplot(gs[3,1])
    ax_unused.axis("off")

    for i, arr2d in enumerate(phase_data_list):
        c = phase_colors[i % len(phase_colors)]
        Nrows, dim = arr2d.shape

        # Y–Z
        if dim>=2:
            ax_xy.plot(arr2d[:,0], arr2d[:,1], color=c, label=f"Phase {i}")

        tvals = np.arange(Nrows)
        if dim>0:
            ax_y.plot(tvals, arr2d[:,0], color=c)
        if dim>1:
            ax_z.plot(tvals, arr2d[:,1], color=c)
        if dim>2:
            ax_phi.plot(tvals, arr2d[:,2], color=c)
        if dim>3:
            ax_r.plot(tvals, arr2d[:,3], color=c)
        if dim>4:
            ax_th.plot(tvals, arr2d[:,4], color=c)

    ax_xy.legend()
    ax_y.grid(True)
    ax_z.grid(True)




