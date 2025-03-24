
from Differential_drive_MPC import DifferentialDriveMPC, DifferentialDriveMPCOptions
import numpy as np
from utils_diff_drive import simulate_closed_loop
from plotting_utils_diff_drive import plot_diff_drive_trajectory

# Compile MPC

mpc_opts = DifferentialDriveMPCOptions()
mpc_opts.N = 100
mpc_opts.step_sizes = [0.05]*100
mpc_opts.switch_stage = 201
# mpc_opts.ellipse_centers = np.array([
#         [-0.5, 0.3],  # Center of first ellipse
#     ])
# mpc_opts.ellipse_half_axes = np.array([
#         [0.01, 0.01],   # Half-axis lengths (a, b) for first ellipse
#     ])
# mpc_opts.box_obstacles = np.array([
#     [-1.1, -0.6, -0.5, 0.5],  # Box with x in [1,2] and y in [3,4]
# ])
mpc = DifferentialDriveMPC(mpc_opts)

print("Acados compiles sucessfully")

# call solve for given initial state
X0 = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  #np.array([1.0, 0.0, 0.0, np.pi, 0.0, 0.0, 0.0])  # Intital state
mpc.solve(X0)
print("Solve runs through succesfully")

# Visualize open loop plan
traj_x, traj_u = mpc.get_planned_trajectory()
print(traj_u)
plot_diff_drive_trajectory(np.zeros(2), mpc=mpc, open_loop_plan=np.array(traj_x))

# Try simulation in closed loop
# duration = 10
# x_traj, u_traj, stage_costs = simulate_closed_loop(X0, mpc, duration)
# print("Closed loop sim runs through succesfully")

# plot_diff_drive_trajectory(np.zeros(2), mpc=mpc, closed_loop_traj=x_traj)

import numpy as np
import matplotlib.pyplot as plt

####################
# 1) Control Input Function
####################

def control_input(t):
    """
    Defines the control input over time.
    Can be modified to have time-varying control input.
    """
    return np.array([1.0, -1.0])  # Modify as needed

####################
# 2) Setup Acados Simulators
####################

def create_sim_solver(model_type, step_size):
    """
    Creates an Acados simulation solver with the given step size.
    model_type: 'full' (with actuation) or 'reduced' (no actuation).
    step_size: integration step size.
    """
    options = DifferentialDriveMPCOptions()
    options.step_sizes = [step_size] * options.N  # Use uniform step size
    options.integrator_type = "IRK"  # Use IRK

    if model_type == "full":
        mpc = DifferentialDriveMPC(options)
        return mpc.acados_sim_solver, 7  # Full model has 7 states
    elif model_type == "reduced":
        mpc = DifferentialDriveMPC(options)
        return mpc.acados_sim_solver_no_act, 5  # Reduced model has 5 states
    else:
        raise ValueError("Unknown model type: Choose 'full' or 'reduced'.")


def simulate_with_acados(sim_solver, X0, T_end, nx):
    """
    Simulates the system forward using AcadosSimSolver.
    sim_solver: Acados simulation solver.
    X0: Initial state (correct size).
    T_end: Simulation duration.
    nx: Number of states for the specific model.
    """
    sim_solver.set("x", np.reshape(X0, (nx, )))  # Ensure correct shape

    t_values = [0.0]
    X_values = [X0]
    step_size = sim_solver.acados_sim.solver_options.T
    num_steps = int(T_end / step_size)

    for i in range(num_steps):
        U = control_input(t_values[-1])  # Get control input at the current time
        sim_solver.set("u", U)

        status = sim_solver.solve()
        if status != 0:
            print(f"Simulation failed with status {status}")
            break

        X_new = sim_solver.get("x")
        X_values.append(X_new)
        t_values.append(t_values[-1] + step_size)
        sim_solver.set("x", X_new)  # Set new initial condition for next step

    return np.array(t_values), np.array(X_values)

####################
# 3) Compute Ground Truth with Small Step Size
####################
T_end = 2.0
X0_full = np.zeros(7)  # Full model (7 states)
X0_reduced = np.zeros(5)  # Reduced model (5 states)

fine_step = 0.001  # Very small step size for ground truth
sim_solver_full, nx_full = create_sim_solver("full", fine_step)
t_fine, X_fine = simulate_with_acados(sim_solver_full, X0_full, T_end, nx_full)
x1_fine, x2_fine = X_fine[:, 0], X_fine[:, 1]

####################
# 4) Simulate With Different Step Sizes & Save Trajectories
####################

step_sizes = np.linspace(0.005, 0.1, 10)  # Different step sizes from 5ms to 100ms
errors_full, errors_reduced = [], []
trajectories_full, trajectories_reduced = {}, {}

for dt in step_sizes:
    # (a) Simulate the full model
    sim_solver_full, nx_full = create_sim_solver("full", dt)
    t_coarse_full, X_coarse_full = simulate_with_acados(sim_solver_full, X0_full, T_end, nx_full)
    del sim_solver_full
    x1_coarse_full, x2_coarse_full = X_coarse_full[:, 0], X_coarse_full[:, 1]
    trajectories_full[dt] = (x1_coarse_full, x2_coarse_full)

    # (b) Simulate the reduced model
    sim_solver_reduced, nx_reduced = create_sim_solver("reduced", dt)
    t_coarse_red, X_coarse_red = simulate_with_acados(sim_solver_reduced, X0_reduced, T_end, nx_reduced)
    del sim_solver_reduced
    x1_coarse_red, x2_coarse_red = X_coarse_red[:, 0], X_coarse_red[:, 1]
    trajectories_reduced[dt] = (x1_coarse_red, x2_coarse_red)

    # (c) Compute L2 error
    sum_sq_error_full = np.sum((x1_coarse_full - x1_fine[:len(x1_coarse_full)])**2 + (x2_coarse_full - x2_fine[:len(x2_coarse_full)])**2)
    sum_sq_error_red = np.sum((x1_coarse_red - x1_fine[:len(x1_coarse_red)])**2 + (x2_coarse_red - x2_fine[:len(x2_coarse_red)])**2)

    errors_full.append(np.sqrt(sum_sq_error_full))
    errors_reduced.append(np.sqrt(sum_sq_error_red))

####################
# 5) Plot Results
####################

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot full model trajectories
for dt, (x1_traj, x2_traj) in trajectories_full.items():
    axs[0].plot(x1_traj, x2_traj, label=f"dt={dt:.3f}")

axs[0].set_xlabel("x1")
axs[0].set_ylabel("x2")
axs[0].set_title("Full Model (7D) Trajectories")
axs[0].legend()
axs[0].grid(True)

# Plot reduced model trajectories
for dt, (x1_traj, x2_traj) in trajectories_reduced.items():
    axs[1].plot(x1_traj, x2_traj, label=f"dt={dt:.3f}")

axs[1].set_xlabel("x1")
axs[1].set_ylabel("x2")
axs[1].set_title("Reduced Model (5D) Trajectories")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
