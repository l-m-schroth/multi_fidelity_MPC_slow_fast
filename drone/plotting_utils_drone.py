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
    A plotting function that handles either:
      1) A multi-phase open-loop plan (open_loop_plan != None), 
         forcibly skipping r(t), theta(t) after switching.
      2) A single-phase closed-loop trajectory (open_loop_plan == None but closed_loop_traj != None)
         that might have 12D states the entire time; we DO NOT cut out r,theta.

    If both open_loop_plan and closed_loop_traj are None => no data is plotted.
    If both are given => we default to open_loop_plan logic.
    """

    # ------------------------ 
    # 0) Decide what data to plot
    # ------------------------
    # If the user gave open_loop_plan, we do multi-phase logic.
    # If not, but closed_loop_traj is given, treat it as single-phase.
    # If neither is given => skip.
    if open_loop_plan is not None:
        # CASE A: multi-phase open-loop plan
        #   We skip r,theta in phases > 0
        data_list = open_loop_plan
        multi_phase = True
    elif closed_loop_traj is not None:
        # CASE B: single-phase closed-loop trajectory (no skipping)
        data_list = [x for x in closed_loop_traj]
        multi_phase = False
    else:
        print("No data to plot (both open_loop_plan and closed_loop_traj are None).")
        return

    # --------------------------
    # If we have no data_list => skip
    # --------------------------
    if not data_list:
        print("Empty data list => nothing to plot.")
        return

    # For dimensional checks
    def unify_snapshots_into_2d(snapshot_list):
        return np.vstack([arr.reshape(1, -1) for arr in snapshot_list if arr is not None])

    # Same for controls
    def unify_controls(u_list):
        # skip None or empty
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

    # -----------------------------------------------------------
    # Partition logic for multi-phase vs single-phase
    # -----------------------------------------------------------
    N = mpc.opts.N
    switch_stage = np.clip(mpc.opts.switch_stage, 0, N)

    if multi_phase:
        # MULTI-PHASE:
        phaseA_list = data_list[:switch_stage+1]
        phaseB_list = data_list[switch_stage+1:]
        phaseA_data = unify_snapshots_into_2d(phaseA_list) if len(phaseA_list)>0 else None
        phaseB_data = unify_snapshots_into_2d(phaseB_list) if len(phaseB_list)>0 else None

        phases_data   = []
        phases_colors = []
        if phaseA_data is not None and phaseA_data.shape[0]>0:
            phases_data.append(phaseA_data)
            phases_colors.append("red")
        if phaseB_data is not None and phaseB_data.shape[0]>0:
            phases_data.append(phaseB_data)
            phases_colors.append("blue")

        # partition controls similarly
        phases_u, phases_u_colors = [], []
        if u_traj is not None:
            switch_u = min(switch_stage, len(u_traj))
            uA_list = u_traj[:switch_u]
            uB_list = u_traj[switch_u:]
            phaseA_u = unify_controls(uA_list)
            phaseB_u = unify_controls(uB_list)

            if phaseA_data is not None and phaseA_data.shape[0]>0:
                if phaseA_u is not None and phaseA_u.shape[0]>0:
                    phases_u.append(phaseA_u)
                    phases_u_colors.append("red")
                else:
                    phases_u.append(None)
                    phases_u_colors.append("red")

            if phaseB_data is not None and phaseB_data.shape[0]>0:
                if phaseB_u is not None and phaseB_u.shape[0]>0:
                    phases_u.append(phaseB_u)
                    # if open_loop_plan => color=blue for second phase
                    phases_u_colors.append("blue")
                else:
                    phases_u.append(None)
                    phases_u_colors.append("blue")

        else:
            # no controls => we have no phases_u
            pass

    else:
        # SINGLE-PHASE (closed-loop or single-phase plan)
        # unify entire data_list into a single array
        single_data = unify_snapshots_into_2d(data_list)
        phases_data = [single_data]
        # single color => red
        phases_colors= ["red"]

        # unify controls if given
        phases_u = []
        phases_u_colors = []
        if u_traj is not None:
            # unify all
            single_u = unify_controls(u_traj)
            if single_u is not None:
                phases_u = [single_u]
                phases_u_colors = ["red"]
            else:
                phases_u = [None]
                phases_u_colors= ["red"]

    # =========== 
    # FIGURE 1: y–z & time subplots
    # ===========
    fig1 = plt.figure(figsize=(10,10))
    gs = fig1.add_gridspec(4, 2, height_ratios=[1.5,1,1,1])
    ax_xy = fig1.add_subplot(gs[0,:])
    ax_xy.set_title(f"{plot_title} - YZ Trajectory")
    ax_xy.set_xlabel("y [m]")
    ax_xy.set_ylabel("z [m]")
    ax_xy.grid(True)
    ax_xy.axis("equal")
    if reference_xy is not None:
        ax_xy.plot(reference_xy[0], reference_xy[1], '*', color='gold', markersize=10, label="Target")

    ax_y   = fig1.add_subplot(gs[1,0]); ax_y.set_title("y(t)");   ax_y.grid(True)
    ax_z   = fig1.add_subplot(gs[1,1]); ax_z.set_title("z(t)");   ax_z.grid(True)
    ax_phi = fig1.add_subplot(gs[2,0]); ax_phi.set_title("phi(t)"); ax_phi.grid(True)
    ax_r   = fig1.add_subplot(gs[2,1]); ax_r.set_title("r(t)");   ax_r.grid(True)
    ax_th  = fig1.add_subplot(gs[3,0]); ax_th.set_title("theta(t)"); ax_th.grid(True)
    ax_unused = fig1.add_subplot(gs[3,1]); ax_unused.axis("off")

    time_offset = 0
    for i, phase_arr in enumerate(phases_data):
        c = phases_colors[i]
        n_s, dim_s = phase_arr.shape
        t_s = np.arange(n_s) + time_offset

        # y-z
        if dim_s>=2:
            ax_xy.plot(phase_arr[:,0], phase_arr[:,1], color=c, label=f"Phase{i}")
        # y,z,phi
        if dim_s>0: ax_y.plot(t_s, phase_arr[:,0], color=c)
        if dim_s>1: ax_z.plot(t_s, phase_arr[:,1], color=c)
        if dim_s>2: ax_phi.plot(t_s, phase_arr[:,2], color=c)

        # For multi-phase => skip r,theta in second phase (i>0). 
        # For single-phase => always plot if dim>4
        if multi_phase:
            # logic => if i==0 and dim_s>4 => plot r,theta
            if i==0 and dim_s>4:
                ax_r.plot(t_s, phase_arr[:,3], color=c)
                ax_th.plot(t_s, phase_arr[:,4], color=c)
        else:
            # single-phase => if dim_s>4 => plot r,theta entire time
            if dim_s>4:
                ax_r.plot(t_s, phase_arr[:,3], color=c)
                ax_th.plot(t_s, phase_arr[:,4], color=c)

        time_offset += n_s

    ax_xy.legend()

    # =========== 
    # FIGURE 2: configuration 
    # ===========
    fig2, ax2 = plt.subplots(figsize=(8,6))
    ax2.set_title(f"{plot_title} - Drone & Pendulum Configuration")
    ax2.set_xlabel("y [m]")
    ax2.set_ylabel("z [m]")
    ax2.grid(True)
    ax2.axis("equal")
    if reference_xy is not None:
        ax2.plot(reference_xy[0], reference_xy[1], '*', color='gold', markersize=10, label="Target")
        ax2.legend()

    L = getattr(mpc.opts, 'L_rot', 0.2)
    for i, phase_arr in enumerate(phases_data):
        c = phases_colors[i]
        n_s, dim_s = phase_arr.shape

        # multi-phase => skip pendulum in second phase
        # single-phase => always plot if dim_s>4
        do_pendulum = False
        if multi_phase:
            if i==0 and dim_s>4:
                do_pendulum = True
        else:
            if dim_s>4:
                do_pendulum = True

        for idx_ in range(0, n_s, step_pose):
            row = phase_arr[idx_]
            y_ = row[0] if dim_s>0 else 0
            z_ = row[1] if dim_s>1 else 0
            phi_ = row[2] if dim_s>2 else 0

            # Drone line
            dy_ = L*np.cos(phi_)
            dz_ = L*np.sin(phi_)
            ax2.plot([y_-dy_, y_+dy_], [z_-dz_, z_+dz_], color='k', linewidth=2)
            ax2.plot(y_, z_, 'ro', markersize=3)

            if do_pendulum:
                r_  = row[3]
                th_ = row[4]
                load_y = y_ + r_*np.sin(th_)
                load_z = z_ - r_*np.cos(th_)
                ax2.plot([y_, load_y],[z_, load_z], color='green', linewidth=2)

    # =========== 
    # FIGURE 3: thrust + input 
    # ===========
    if u_traj is not None:
        fig3 = plt.figure(figsize=(8,6))
        gs3 = fig3.add_gridspec(2,1)
        ax_thrust = fig3.add_subplot(gs3[0,0])
        ax_thrust.set_title("Left & Right Thrust vs. time")
        ax_thrust.grid(True)

        ax_inputs = fig3.add_subplot(gs3[1,0])
        ax_inputs.set_title("Control signals (dw1,dw2) or (F1,F2)")
        ax_inputs.grid(True)

        time_offset_u = 0
        for i, phase_arr in enumerate(phases_data):
            color_ = phases_colors[i]
            if i<len(phases_u):
                u2d = phases_u[i]
                # color for inputs
                c_u = phases_u_colors[i] if i<len(phases_u_colors) else color_
            else:
                u2d, c_u = None, color_

            n_s, ds = phase_arr.shape
            t_s = np.arange(n_s) + time_offset_u

            if u2d is not None:
                nu, du_ = u2d.shape
                t_u = np.arange(nu) + time_offset_u
            else:
                nu, du_ = 0,0
                t_u = []

            c_ = getattr(mpc.opts, 'c', 1.0)

            # if ds>=12 => w1= x[:,10], w2= x[:,11]
            # if ds>=8 => w1= x[:,6], x[:,7]
            # else => direct thrust => from input
            if ds>=12:
                w1 = phase_arr[:,10]
                w2 = phase_arr[:,11]
                thr_l = c_*w1
                thr_r = c_*w2
                ax_thrust.plot(t_s, thr_l, color=color_, label=f"Ph{i}: left thr")
                ax_thrust.plot(t_s, thr_r, color=color_, linestyle='--', label=f"Ph{i}: right thr")
            elif ds>=8:
                w1 = phase_arr[:,6]
                w2 = phase_arr[:,7]
                thr_l = c_*w1
                thr_r = c_*w2
                ax_thrust.plot(t_s, thr_l, color=color_, label=f"Ph{i}: left thr")
                ax_thrust.plot(t_s, thr_r, color=color_, linestyle='--', label=f"Ph{i}: right thr")
            else:
                # direct thrust => from input
                if (u2d is not None) and du_>=2:
                    F1 = u2d[:,0]
                    F2 = u2d[:,1]
                    ax_thrust.plot(t_u, F1, color=color_, label=f"Ph{i}: F1")
                    ax_thrust.plot(t_u, F2, color=color_, linestyle='--', label=f"Ph{i}: F2")

            # inputs
            if (u2d is not None) and du_>=2:
                ax_inputs.plot(t_u, u2d[:,0], color=c_u, label=f"Ph{i}: u0")
                ax_inputs.plot(t_u, u2d[:,1], color=c_u, linestyle='--', label=f"Ph{i}: u1")

            time_offset_u += max(n_s, nu)

        ax_thrust.legend()
        ax_inputs.legend()

    plt.show()





