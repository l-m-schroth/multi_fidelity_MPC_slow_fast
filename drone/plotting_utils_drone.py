import numpy as np
import matplotlib.pyplot as plt

def plot_drone_mpc_solution(
    mpc, 
    reference_xy=None,
    closed_loop_traj=None,
    open_loop_plan=None,
    u_traj=None,
    plot_title="Drone + Pendulum MPC",
    step_pose=10
):
    """
    Updated plotting function for the new DroneMPC class logic:
      - Full model (12D) uses [y,z,phi,r,theta,..., w1,w2]
      - Approx model (10D) uses [y,z,phi,theta,..., w1,w2]
      - Possibly a transition model (10D or similar).
      - We color each phase differently if multi-phase:
         Phase 0 -> red, Phase 1 -> green, Phase 2 -> blue
      - For the geometry plot (Figure 2), if dimension=12, we interpret
         r = x[3], theta = x[4].
        If dimension=10, we interpret a fixed length l0 from mpc.opts.l0,
         theta = x[3].
      - The time subplots show y,z,phi always if present; we show r(t) and/or
         theta(t) only if that dimension is valid.

    If open_loop_plan is None but closed_loop_traj is given, we assume
    single-phase data. If multi-phase, we do a partition by switch_stage
    or the actual length of the plan (like your older code).
    """

    # ------------------------------------------------
    # 1) Decide data source: multi-phase open_loop_plan
    #    vs single-phase closed_loop_traj
    # ------------------------------------------------
    if open_loop_plan is not None:
        data_list = open_loop_plan
        multi_phase = True
    elif closed_loop_traj is not None:
        data_list = [row for row in closed_loop_traj]
        multi_phase = False
    else:
        print("No data to plot (both open_loop_plan and closed_loop_traj are None).")
        return

    if not data_list:
        print("Empty data list => nothing to plot.")
        return

    # ------------------------------------------------
    # Helpers to unify snapshots & controls
    # ------------------------------------------------
    def unify_snapshots_into_2d(snapshot_list):
        filtered = []
        for arr in snapshot_list:
            if arr is None:
                continue
            arr2 = np.asarray(arr)
            if arr2.size == 0:
                continue
            if arr2.ndim == 1:
                arr2 = arr2.reshape(1, -1)
            filtered.append(arr2)
        if len(filtered) == 0:
            return None
        return np.vstack(filtered)

    def unify_controls(u_list):
        filtered = []
        for item in u_list:
            if item is None:
                continue
            arr = np.asarray(item)
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            filtered.append(arr)
        if len(filtered)==0:
            return None
        return np.vstack(filtered)

    # We'll figure out if we have up to 3 phases for multi-phase:
    N = mpc.opts.N
    switch_stage = np.clip(mpc.opts.switch_stage, 0, N)

    # ------------------------------------------------
    # 2) Partition states for multi-phase vs single-phase
    # ------------------------------------------------
    if multi_phase:
        # Possibly up to 3 phases:
        #  - Phase 0: [0..switch_stage]
        #  - Phase 1: maybe the discrete transition at switch_stage
        #  - Phase 2: from switch_stage+1..end
        # This is similar to your old code
        phaseA_list = data_list[:switch_stage+1]
        phaseB_list = data_list[switch_stage+1:]
        phaseA_data = unify_snapshots_into_2d(phaseA_list)
        phaseB_data = unify_snapshots_into_2d(phaseB_list)

        # If you have a middle "Phase 1" (like a single discrete stage),
        # you can separate it out or keep to 2 phases. We'll do 2-phase for simplicity,
        # or 3-phase if you want:
        # If you do have a discrete transition as an extra index, you can color it green.
        # But in your code, you mention you do "N0, 1, N2" for the solver. Let's do that:

        # We do "phases_data" logic like your old approach. For 2-phase or 3-phase:
        # We'll check if there's a single "discrete" index.
        # Actually, let's keep it simpler and produce 2-phase:
        phases_data   = []
        phases_colors = []
        if phaseA_data is not None and phaseA_data.shape[0]>0:
            phases_data.append(phaseA_data)
            phases_colors.append("red")
        if phaseB_data is not None and phaseB_data.shape[0]>0:
            phases_data.append(phaseB_data)
            phases_colors.append("blue")

        # controls
        phases_u = []
        phases_u_colors = []
        if u_traj is not None:
            # We have N control intervals => up to switch_stage => phase 0, then after => phase 1
            switch_u = min(switch_stage, len(u_traj))
            uA_list = u_traj[:switch_u]
            uB_list = u_traj[switch_u:]
            phaseA_u = unify_controls(uA_list)
            phaseB_u = unify_controls(uB_list)

            if phaseA_data is not None and phaseA_data.shape[0]>0:
                phases_u.append(phaseA_u)
                phases_u_colors.append("red")
            if phaseB_data is not None and phaseB_data.shape[0]>0:
                phases_u.append(phaseB_u)
                phases_u_colors.append("blue")
        else:
            phases_u = []
            phases_u_colors = []

    else:
        # single-phase => unify entire data_list
        single_data = unify_snapshots_into_2d(data_list)
        phases_data   = []
        phases_colors = []
        if single_data is not None and single_data.shape[0]>0:
            phases_data.append(single_data)
            phases_colors.append("red")

        # unify controls if given
        phases_u = []
        phases_u_colors= []
        if u_traj is not None:
            single_u = unify_controls(u_traj)
            phases_u.append(single_u)
            phases_u_colors.append("red")
        else:
            phases_u = [None]
            phases_u_colors= ["red"]

    # ------------------------------------------------
    # FIGURE 1: (y,z) + time subplots
    # ------------------------------------------------
    fig1 = plt.figure(figsize=(10,10))
    gs = fig1.add_gridspec(4, 2, height_ratios=[1.5,1,1,1])
    ax_xy = fig1.add_subplot(gs[0,:])
    ax_xy.set_title(f"{plot_title} - YZ Trajectory")
    ax_xy.set_xlabel("y [m]")
    ax_xy.set_ylabel("z [m]")
    ax_xy.grid(True)
    ax_xy.axis("equal")

    if reference_xy is not None:
        ax_xy.plot(reference_xy[0], reference_xy[1], marker='*', color='gold', markersize=10, label="Target")

    ax_y   = fig1.add_subplot(gs[1,0]); ax_y.set_title("y(t)");   ax_y.grid(True)
    ax_z   = fig1.add_subplot(gs[1,1]); ax_z.set_title("z(t)");   ax_z.grid(True)
    ax_phi = fig1.add_subplot(gs[2,0]); ax_phi.set_title("phi(t)");ax_phi.grid(True)
    ax_r   = fig1.add_subplot(gs[2,1]); ax_r.set_title("r(t)");   ax_r.grid(True)
    ax_th  = fig1.add_subplot(gs[3,0]); ax_th.set_title("theta(t)"); ax_th.grid(True)
    ax_unused= fig1.add_subplot(gs[3,1]); ax_unused.axis("off")

    time_offset = 0
    for i, phase_arr in enumerate(phases_data):
        color_ = phases_colors[i]
        n_s, dim_s = phase_arr.shape
        t_s = np.arange(n_s) + time_offset

        # Plot (y,z)
        if dim_s>=2:
            ax_xy.plot(phase_arr[:,0], phase_arr[:,1], color=color_, label=f"Phase{i} (dim={dim_s})")
        # Plot y,z,phi
        if dim_s>0:
            ax_y.plot(t_s, phase_arr[:,0], color=color_)
        if dim_s>1:
            ax_z.plot(t_s, phase_arr[:,1], color=color_)
        if dim_s>2:
            ax_phi.plot(t_s, phase_arr[:,2], color=color_)

        # If dimension=12 => we have r at index=3, theta=4
        # If dimension=10 => we have theta at index=3, no r
        # we only plot r(t) if dim==12
        if dim_s==12:
            ax_r.plot(t_s, phase_arr[:,3], color=color_)
            ax_th.plot(t_s, phase_arr[:,4], color=color_)
        elif dim_s==10:
            # no r => only plot theta(t)
            ax_th.plot(t_s, phase_arr[:,3], color=color_)

        time_offset += n_s

    ax_xy.legend()

    # ------------------------------------------------
    # FIGURE 2: configuration plot
    # ------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8,6))
    ax2.set_title(f"{plot_title} - Drone & Pendulum Configuration")
    ax2.set_xlabel("y [m]")
    ax2.set_ylabel("z [m]")
    ax2.grid(True)
    ax2.axis("equal")

    if reference_xy is not None:
        ax2.plot(reference_xy[0], reference_xy[1], marker='*', color='gold', markersize=10, label="Target")
        ax2.legend()

    l0 = mpc.opts.l0
    L  = mpc.opts.L_rot

    # We'll step through each phase. For dimension=12 => use r=x[3], theta=x[4].
    # For dimension=10 => use r=l0, theta=x[3].
    # If dimension<10 => skip any pendulum line.
    time_offset_conf = 0
    for i, phase_arr in enumerate(phases_data):
        color_ = phases_colors[i]
        n_s, dim_s = phase_arr.shape

        for idx_ in range(0, n_s, step_pose):
            row = phase_arr[idx_]
            y_   = row[0] if dim_s>0 else 0
            z_   = row[1] if dim_s>1 else 0
            phi_ = row[2] if dim_s>2 else 0

            # Draw drone body => from (y_,z_) in direction phi
            dy_ = L*np.cos(phi_)
            dz_ = L*np.sin(phi_)
            ax2.plot([y_-dy_, y_+dy_],[z_-dz_, z_+dz_], color='k', linewidth=2)
            ax2.plot(y_, z_, 'ro', markersize=3)

            # Pendulum line
            if dim_s==12:
                # r=x[3], theta=x[4]
                r_   = row[3]
                th_  = row[4]
                load_y = y_ + r_*np.sin(th_)
                load_z = z_ - r_*np.cos(th_)
                ax2.plot([y_, load_y],[z_, load_z], color='green', linewidth=2)
            elif dim_s==10:
                # fixed length = l0, angle= x[3]
                th_ = row[3]
                load_y = y_ + l0*np.sin(th_)
                load_z = z_ - l0*np.cos(th_)
                ax2.plot([y_, load_y],[z_, load_z], color='green', linewidth=2)
            # else <10 => skip

        time_offset_conf += n_s

    # ------------------------------------------------
    # FIGURE 3: Thrust & input signals
    # ------------------------------------------------
    if u_traj is not None:
        fig3 = plt.figure(figsize=(8,6))
        gs3 = fig3.add_gridspec(2,1)
        ax_thrust= fig3.add_subplot(gs3[0,0])
        ax_thrust.set_title("Left & Right Thrust vs. time")
        ax_thrust.grid(True)

        ax_inputs= fig3.add_subplot(gs3[1,0])
        ax_inputs.set_title("Control signals (dw1,dw2)")
        ax_inputs.grid(True)

        time_offset_u = 0
        for i, phase_arr in enumerate(phases_data):
            c = phases_colors[i]
            if i<len(phases_u):
                u2d = phases_u[i]
                cu  = phases_u_colors[i]
            else:
                u2d, cu = None, c

            n_s, ds = phase_arr.shape
            t_s = np.arange(n_s) + time_offset_u
            if u2d is not None:
                nu, du_ = u2d.shape
                t_u = np.arange(nu) + time_offset_u
            else:
                nu, du_ = 0,0
                t_u = []

            c_ = getattr(mpc.opts, 'c', 1.0)

            # If ds >=12 => w1= x[:,10], x[:,11]
            # If ds=10 => w1= x[:,8], x[:,9]
            # else => skip or direct thrust => not used in new code, but let's keep it
            if ds>=12:
                w1 = phase_arr[:,10]
                w2 = phase_arr[:,11]
                thr_l = c_*w1
                thr_r = c_*w2
                ax_thrust.plot(t_s, thr_l, color=c, label=f"Ph{i}: left thr")
                ax_thrust.plot(t_s, thr_r, color=c, linestyle='--', label=f"Ph{i}: right thr")
            elif ds==10:
                w1 = phase_arr[:,8]
                w2 = phase_arr[:,9]
                thr_l = c_*w1
                thr_r = c_*w2
                ax_thrust.plot(t_s, thr_l, color=c, label=f"Ph{i}: left thr")
                ax_thrust.plot(t_s, thr_r, color=c, linestyle='--', label=f"Ph{i}: right thr")
            else:
                # direct thrust? skip or read from input if du_>=2
                if (u2d is not None) and du_>=2:
                    F1 = u2d[:,0]
                    F2 = u2d[:,1]
                    ax_thrust.plot(t_u, F1, color=c, label=f"Ph{i}: F1")
                    ax_thrust.plot(t_u, F2, color=c, linestyle='--', label=f"Ph{i}: F2")

            # plot input signals
            # dimension of input is 2 => [dw1,dw2]
            if (u2d is not None) and du_>=2:
                ax_inputs.plot(t_u, u2d[:,0], color=cu, label=f"Ph{i}: u0")
                ax_inputs.plot(t_u, u2d[:,1], color=cu, linestyle='--', label=f"Ph{i}: u1")

            time_offset_u += max(n_s, nu)

        ax_thrust.legend()
        ax_inputs.legend()

    plt.show()






