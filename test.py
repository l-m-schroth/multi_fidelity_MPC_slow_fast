
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