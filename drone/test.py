from drone_MPC import DroneMPC, DroneMPCOptions
import numpy as np

# %% Hovering test for DroneMPC
# compute needed w such that hovering is even possible
w_eq = 2.1 * 9.81 / 2 

# Create an instance of the MPC options (using the updated physical parameters)
opts = DroneMPCOptions(
    M=2.0,         # drone mass in kg
    m=0.1,         # load mass in kg (small compared to drone)
    Ixx=0.05,      # moment of inertia in kg·m²
    g=9.81,        # gravitational acceleration in m/s²
    c=1.0,         # rotor thrust constant
    L_rot=0.2,     # half distance between rotors in m
    k=1.0,       # stiff spring (N/m)
    l0=0.3,        # spring rest length in m
    # Constraints and cost weighting matrices remain as defined by default

    N = 20,
    switch_stage = 20,
    step_sizes = [0.001]*20,

    w_min = -20,
    w_max = 20,
    w_dot_min = -10,
    w_dot_max = 10,
    phi_min = -10,
    phi_max = 10,
    F_min = -20,
    F_max = 20,
)

# Create an instance of the DroneMPC class
mpc = DroneMPC(opts)

# Define an initial state corresponding to hovering.
# For the full model, the state is:
# [y, z, phi, r, theta, y_dot, z_dot, phi_dot, r_dot, theta_dot, w1, w2]
# where:
#   - y = 0.0 (horizontal position), z = 1.0 (altitude)
#   - phi (drone pitch) = 0.0, theta (pendulum angle) = 0.0
#   - r is set to the spring rest length (l0)
#   - All velocities are 0.
#   - Rotor speeds (w1, w2) are chosen such that the total thrust equals the weight.
l0 = opts.l0
M_total = opts.M + opts.m
# Required total thrust T = (M_total)*g, and assuming symmetric rotors, each rotor speed is set to T/2 (with c=1)
w_val = (M_total * opts.g) / 2.0

x0 = np.array([
    0.0,    # y position
    1.0,    # z position (hover altitude)
    0.0,    # phi (drone pitch)
    l0,     # r (pendulum length set to rest length)
    0.0,    # theta (pendulum angle)
    0.0,    # y_dot
    0.0,    # z_dot
    0.0,    # phi_dot
    0.0,    # r_dot
    0.0,    # theta_dot
    w_val,  # w1 (rotor speed)
    w_val   # w2 (rotor speed)
])
print("Initial hovering state:", x0)

# Define a reference position far away (for example, [10, 10])
pos_ref = np.array([1.0, 1.0])

# set initial guess to avoid division by zero for r
mpc.set_initial_guess(x0, u_guess=np.zeros(2))

# Call the MPC solve function to compute the first control input
u0 = mpc.solve(x0, pos_ref)
print("First control input:", u0)