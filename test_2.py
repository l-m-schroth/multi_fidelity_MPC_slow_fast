import numpy as np
import matplotlib.pyplot as plt

# Define system parameters
mu = 1.0
epsilon_1, epsilon_2, epsilon_3, epsilon_4 = 10, 0.3, 5, 0.02
d1, d2, d3 = 0.4, 0.4, 5

# Define periodic control input u
def control_input(t, scale=0.3):
    return scale*np.sin(2 * np.pi * t), scale*np.sin(2 * np.pi * t) # Example: sinusoidal input

# Define the modified Van der Pol oscillator
def van_der_pol(t, X):
    x1, x2, x3 = X
    u_1, u_2 = control_input(t)
    dx1 = x2 + d1 * u_1 + epsilon_1 * x3
    dx2 = mu * (1 - x1**2) * x2 - x1 + epsilon_2 * x2 + epsilon_3 * x3 + d2 * u_2
    dx3 = - (1 / epsilon_4) * x3 + d3 * u_2
    return np.array([dx1, dx2, dx3])

# Fixed-step RK4 integrator
def rk4_integrate(f, X0, t_span, step_size):
    t_values = np.arange(t_span[0], t_span[1], step_size)
    X_values = np.zeros((len(t_values), len(X0)))
    
    X = np.array(X0)
    for i, t in enumerate(t_values):
        X_values[i] = X
        
        k1 = step_size * f(t, X)
        k2 = step_size * f(t + step_size / 2, X + k1 / 2)
        k3 = step_size * f(t + step_size / 2, X + k2 / 2)
        k4 = step_size * f(t + step_size, X + k3)
        
        X += (k1 + 2 * k2 + 2 * k3 + k4) / 6  # RK4 update rule

    return t_values, X_values

# Initial conditions
X0 = [1.0, 0.0, 0.0]
t_span = (0, 50)  # Time range

# Different step sizes for integration
step_size = 0.008  # Decreasing step sizes, RUKU fails for 0.03

plt.figure(figsize=(8, 6))

# Solve the ODE using RK4 with a fixed step size
t_vals, X_vals = rk4_integrate(van_der_pol, X0, t_span, step_size)

# Extract solutions
x1, x2, x3 = X_vals[:, 0], X_vals[:, 1],  X_vals[:, 2]

# Plot trajectories
plt.plot(x1, x2, label=f"Step size: {step_size}")

# Plot settings
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Van der Pol Oscillator - Phase Space Trajectories (Fixed Step Size RK4)")
plt.legend()
plt.grid()
plt.show()

# Plot the the fast state over time
plt.plot(t_vals, x3)
plt.xlabel("$t$")
plt.ylabel("$x_3$")
plt.title("Van der Pol Oscillator - Fast state development (Fixed Step Size RK4)")
plt.legend()
plt.grid()
plt.show





from plotting_utils import plot_phase_space

def Phi_t(t):
    omega_1 = 1.3  # Base ellipse frequency
    alpha_1 = 2  # Major axis length
    alpha_2 = 1.5  # Minor axis length
    
    beta = 0.3  # Oscillation magnitude
    nu = 6 * omega_1  # Frequency of normal oscillation
    
    # Base ellipse
    x1 = alpha_1 * np.sin(omega_1 * t)
    x2 = alpha_2 * np.cos(omega_1 * t)
    
    # Tangent vector
    Tx = alpha_1 * omega_1 * np.cos(omega_1 * t)
    Ty = -alpha_2 * omega_1 * np.sin(omega_1 * t)
    
    # Normal vector (rotated tangent)
    Nx = alpha_2 * omega_1 * np.sin(omega_1 * t)
    Ny = alpha_1 * omega_1 * np.cos(omega_1 * t)
    
    # Normalize the normal vector
    norm = np.sqrt(Nx**2 + Ny**2)
    Nx /= norm
    Ny /= norm
    
    # Add oscillation in the normal direction
    x1_perturbed = x1 + beta * Nx * np.sin(nu * t)
    x2_perturbed = x2 + beta * Ny * np.sin(nu * t)
    
    return x1_perturbed, x2_perturbed


plot_phase_space(Phi_t)


from Van_der_pol_MPC import VanDerPolMPC, VanDerPolMPCOptions
from copy import deepcopy
from utils import simulate_closed_loop
from utils_diff_drive import compute_exponential_step_sizes

# Parameters for experiment
t0 = 0.0
x01, x02 = Phi_t(t=t0) # Set initial state on the trajectory
x03 = 0.0
x0 = np.vstack((x01, x02, x03))

mpc_opts_base = VanDerPolMPCOptions(
    # dynamics
    mu=mu,
    epsilon_1=epsilon_1,
    epsilon_2=epsilon_2,
    epsilon_3=epsilon_3,
    epsilon_4=epsilon_4,
    d1=d1,
    d2=d2,
    d3=d3,

    # Define two ellipses to avoid
    ellipse_centers=np.array([
        #[-1.75, 1.1],  # Center of first ellipse
        [1.9, -0.4],   # Center of second ellipse
        [1.4, -0.4],   # Center of second ellipse
        [2.4, -0.4]   # Center of second ellipse
    ]),
    ellipse_half_axes=np.array([
        #[0.3, 0.3],   # Half-axis lengths (a, b) for first ellipse
        [0.6, 0.2],    # Half-axis lengths (a, b) for second ellipse
        [0.2, 0.5],    # Half-axis lengths (a, b) for second ellipse
        [0.2, 0.5],    # Half-axis lengths (a, b) for second ellipse
    ])
)

### Create Simulator, model integrated with small step size treated as ground truth ###
dt_sim = 0.0025
mpc_opts_sim_solver = deepcopy(mpc_opts_base)
mpc_opts_sim_solver.N = 1
mpc_opts_sim_solver.step_size_list = [dt_sim]
mpc_opts_sim_solver.switch_stage = 10
mpc_opts_sim_solver.integrator_type = "IRK"
mpc_sim_solver = VanDerPolMPC(mpc_opts_sim_solver, Phi_t)
sim_solver = mpc_sim_solver.acados_sim_solver_3d

# Simulation parameter
duration = 10

### Baseline MPC ###
mpc_opts_baseline = deepcopy(mpc_opts_base)
N = 100 #150 fails, 250 works, ERK fails
mpc_opts_baseline.N = N
test = compute_exponential_step_sizes(0.005, 250*0.005, N)
mpc_opts_baseline.step_size_list = test#[0.005] * N
mpc_opts_baseline.integrator_type = "IRK"
mpc_opts_baseline.switch_stage = 25 # no switching, use exact model
mpc_baseline = VanDerPolMPC(mpc_opts_baseline, Phi_t)

_ = mpc_baseline.solve(x0, t0)
plot_phase_space(Phi_t, mpc_baseline, open_loop_plan=True)



# # Run closed-loop simulation
# x_traj, u_traj, costs = simulate_closed_loop(
#     x0, mpc_baseline, duration, sigma_noise=0.0, 
#     sim_solver=sim_solver, control_step=2, plot_open_loop_plan=False)
# print("Baseline mean costs:", np.mean(costs))
# plot_phase_space(Phi_t, mpc_baseline, closed_loop_traj=x_traj)