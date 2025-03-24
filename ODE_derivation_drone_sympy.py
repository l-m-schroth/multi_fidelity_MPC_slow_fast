
###############################################################################
# 2D QUADROTOR + SPRING PENDULUM SYMBOLIC MODEL
# 
# This script sets up:
#   - Drone position (y, z) and pitch angle (phi)
#   - Spring pendulum (r, theta)
#   - Rotor spin rates (w1, w2), with control inputs (u1 = dw1/dt, u2 = dw2/dt)
#   - External forces on drone (f_ext_d) and mass (f_ext_m)
#
# Then it derives the Lagrangian-based EOM for (y, z, phi, r, theta),
# appends w1_dot and w2_dot, and finally exports to a CasADi function.
###############################################################################

import sympy
import os

# --- Directory setup (adjust path as needed) ---
save_dir = os.path.dirname(__file__)
os.makedirs(save_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 1) Define symbols
# -----------------------------------------------------------------------------
t = sympy.Symbol('t', real=True)  # time symbol

# Drone + pendulum parameters
M   = sympy.Symbol('M',     real=True)  # mass of the drone
m   = sympy.Symbol('m',     real=True)  # mass of the pendulum bob
Ixx = sympy.Symbol('Ixx',   real=True)  # moment of inertia of drone about CoM
g   = sympy.Symbol('g',     real=True)  # gravitational acceleration
k   = sympy.Symbol('k',     real=True)  # spring constant
l0  = sympy.Symbol('l0',    real=True)  # spring rest length
c   = sympy.Symbol('c',     real=True)  # rotor thrust coefficient
L   = sympy.Symbol('L',     real=True)  # half-distance between rotors (lever arm)

# External forces (drone, mass) in the plane
f_ext_dy = sympy.Symbol('f_ext_dy', real=True)
f_ext_dz = sympy.Symbol('f_ext_dz', real=True)
f_ext_my = sympy.Symbol('f_ext_my', real=True)
f_ext_mz = sympy.Symbol('f_ext_mz', real=True)

# Control inputs (u1 = dw1/dt, u2 = dw2/dt)
u1 = sympy.Symbol('u1', real=True)
u2 = sympy.Symbol('u2', real=True)

# Generalized coordinates as functions of time
y     = sympy.Function('y')(t)       # drone horizontal position
z     = sympy.Function('z')(t)       # drone vertical position
phi   = sympy.Function('phi')(t)     # drone pitch angle
r     = sympy.Function('r')(t)       # pendulum spring length
theta = sympy.Function('theta')(t)   # pendulum angle from vertical (downward)

# Rotor spin rates as state variables
w1 = sympy.Function('w1')(t)
w2 = sympy.Function('w2')(t)

# Time derivatives
y_dot     = y.diff(t)
z_dot     = z.diff(t)
phi_dot   = phi.diff(t)
r_dot     = r.diff(t)
theta_dot = theta.diff(t)
w1_dot    = w1.diff(t)
w2_dot    = w2.diff(t)

# -----------------------------------------------------------------------------
# 2) Express Kinetic Energy (T)
# -----------------------------------------------------------------------------
# 2A) Drone's kinetic energy (translation + rotation)
T_drone = sympy.Rational(1,2)*M*(y_dot**2 + z_dot**2) \
          + sympy.Rational(1,2)*Ixx*(phi_dot**2)

# 2B) Pendulum bob's position in inertial frame
#     If theta=0 means "straight down", we set:
#        y_m = y + r*sin(theta)
#        z_m = z - r*cos(theta)
y_m = y + r*sympy.sin(theta)
z_m = z - r*sympy.cos(theta)

# Kinetic energy of the pendulum bob
y_m_dot = y_m.diff(t)
z_m_dot = z_m.diff(t)
T_pend  = sympy.Rational(1,2)*m*(y_m_dot**2 + z_m_dot**2)

T_total = T_drone + T_pend

# -----------------------------------------------------------------------------
# 3) Potential Energy (U)
# -----------------------------------------------------------------------------
# Drone’s gravitational potential
U_drone = M*g*z

# Pendulum bob gravitational potential
U_mass = m*g*z_m

# Spring potential: 1/2 k (r - l0)^2
U_spring = sympy.Rational(1,2)*k*(r - l0)**2

U_total = U_drone + U_mass + U_spring

# -----------------------------------------------------------------------------
# 4) Lagrangian
# -----------------------------------------------------------------------------
L = T_total - U_total

# -----------------------------------------------------------------------------
# 5) Non-conservative / Generalized Forces
# -----------------------------------------------------------------------------
# The rotor thrust f = c*(w1 + w2), direction = (-sin(phi), cos(phi))
# The rotor torque tau = c*L*(w2 - w1) about the drone's center (affects phi)

f_thrust  = c*(w1 + w2)
tau_thrust = c*L*(w2 - w1)

# External forces on drone: (f_ext_dy, f_ext_dz)
# External forces on pendulum mass: (f_ext_my, f_ext_mz)

# For Lagrange's eqs: Q_j = Sum of [F_ext . (∂x/∂q_j)]
# We'll build them for the 5 generalized coords: y, z, phi, r, theta

# --- Qy: force in the y-direction on the drone + mass
Qy_drone = f_ext_dy + f_thrust*(-sympy.sin(phi))   # thrust + ext on drone
Qy_mass  = f_ext_my                                # from mass's external force 
#    partial derivative wrt y => mass moves 1:1 with y => dot product = f_ext_my, y_mass = y + rsin(phi)!
Qy_total = Qy_drone + Qy_mass

# --- Qz: force in the z-direction on the drone + mass
Qz_drone = f_ext_dz + f_thrust*sympy.cos(phi)      # thrust + ext on drone
Qz_mass  = f_ext_mz
Qz_total = Qz_drone + Qz_mass

# --- Qphi: torque about drone's CoM from thrust + external
#    The external force at CoM (f_ext_d) has zero moment about CoM (no arm),
#    so Qphi from drone external force is 0 unless an offset is given.
#    We do have rotor torque:
Qphi_total = tau_thrust

#    For mass external force, the pivot is the drone’s CoM, so no offset => 0.

# --- Qr: partial derivative wrt 'r'
#    The external force on the mass is f_ext_m = (f_ext_my, f_ext_mz).
#    Pendulum bob position (y_m, z_m) depends on r => ∂(y_m,z_m)/∂r
#          = ( sin(theta), -cos(theta) )
Qr_mass = f_ext_my*sympy.sin(theta) + f_ext_mz*(-sympy.cos(theta))
#    Drone external force and thrust do not depend on r. 
Qr_total = Qr_mass

# --- Qtheta: partial derivative wrt 'theta'
#     (y_m, z_m) = (y + r sin(theta), z - r cos(theta))
#     ∂(y_m,z_m)/∂theta = (r cos(theta), r sin(theta))
Qtheta_mass = f_ext_my*(r*sympy.cos(theta)) + f_ext_mz*(r*sympy.sin(theta))
Qtheta_total = Qtheta_mass

# -----------------------------------------------------------------------------
# 6) Build Lagrange’s Equations for y, z, phi, r, theta
#    d/dt(∂L/∂qdot) - ∂L/∂q = Q_q
# -----------------------------------------------------------------------------
coords     = [y,     z,     phi,     r,     theta]
coords_dot = [y_dot, z_dot, phi_dot, r_dot, theta_dot]
Q_list     = [Qy_total, Qz_total, Qphi_total, Qr_total, Qtheta_total]

dL_dq     = [L.diff(q)     for q     in coords    ]
dL_dq_dot = [L.diff(q_dot) for q_dot in coords_dot]

ddt_dL_dq_dot = []
for i in range(len(coords)):
    tmp = dL_dq_dot[i].diff(t)
    # For the derivative w.r.t t, explicitly substitute q'(t)-> coords_dot[i]
    # so that e.g. diff(y(t), t) -> y_dot(t) becomes a direct symbol
    subs_dict = {coords_dot[j].diff(t): sympy.Symbol(f'q_ddot[{j}]')
                 for j in range(len(coords))}
    subs_dict.update({coords_dot[j]: sympy.Symbol(f'q_dot[{j}]')
                      for j in range(len(coords))})
    subs_dict.update({coords[j]: sympy.Symbol(f'q[{j}]')
                      for j in range(len(coords))})
    ddt_dL_dq_dot.append(tmp.subs(subs_dict))

# EOM (implicit form): d/dt(∂L/∂qdot) - ∂L/∂q - Q = 0
eom_implicit = []
for i in range(len(coords)):
    term1 = ddt_dL_dq_dot[i]
    term2 = dL_dq[i].subs({coords[j]: sympy.Symbol(f'q[{j}]')
                           for j in range(len(coords))})
    # Q_list[i] also depends on phi, etc. so do the same kind of substitution
    Q_sub = Q_list[i].subs({
        y: sympy.Symbol('q[0]'), z: sympy.Symbol('q[1]'),
        phi: sympy.Symbol('q[2]'), r: sympy.Symbol('q[3]'),
        theta: sympy.Symbol('q[4]'),
        y_dot: sympy.Symbol('q_dot[0]'), z_dot: sympy.Symbol('q_dot[1]'),
        phi_dot: sympy.Symbol('q_dot[2]'), r_dot: sympy.Symbol('q_dot[3]'),
        theta_dot: sympy.Symbol('q_dot[4]'),
        w1: sympy.Symbol('w[0]'), w2: sympy.Symbol('w[1]')
    })
    eom_implicit.append(term1 - term2 - Q_sub)

# -----------------------------------------------------------------------------
# 7) Now handle rotor spin rates w1, w2
#    We have: w1_dot = u1,  w2_dot = u2
#    => eom_w1 = w1_dot - u1 = 0  => w1_ddot doesn't appear.
# -----------------------------------------------------------------------------
# We'll add them as 2 more ODEs in the final system:
eom_w1 = w1_dot - u1
eom_w2 = w2_dot - u2

# Convert them to the same style of "implicit = 0" expression
# (with substituted symbol names for consistent output)
eom_w1_sub = eom_w1.subs({
    w1_dot: sympy.Symbol('w_dot[0]'),
    u1: sympy.Symbol('u[0]'),
    w1: sympy.Symbol('w[0]'),
    w2: sympy.Symbol('w[1]')  # no effect but for consistency
})
eom_w2_sub = eom_w2.subs({
    w2_dot: sympy.Symbol('w_dot[1]'),
    u2: sympy.Symbol('u[1]'),
    w1: sympy.Symbol('w[0]'),
    w2: sympy.Symbol('w[1]')
})

# -----------------------------------------------------------------------------
# 8) Collect final EOM expressions
# -----------------------------------------------------------------------------
eom_all = eom_implicit + [eom_w1_sub, eom_w2_sub]
eom_all_matrix = sympy.Matrix(eom_all)

# Optionally simplify
eom_all_simpl = [sympy.simplify(expr) for expr in eom_all_matrix]

# -----------------------------------------------------------------------------
# 9) Convert to a form suitable for saving to a .py file (CasADi-friendly)
# -----------------------------------------------------------------------------
# Replace trig functions with "ca.sin", "ca.cos", etc.
eom_str_list = []
for eq in eom_all_simpl:
    eq_str = str(eq)
    eq_str = eq_str.replace('sin', 'ca.sin') \
                   .replace('cos', 'ca.cos') \
                   .replace('tan', 'ca.tan') \
                   .replace('asin', 'ca.asin') \
                   .replace('acos', 'ca.acos') \
                   .replace('atan', 'ca.atan') \
                   .replace('exp', 'ca.exp')
    eom_str_list.append(eq_str)

# -----------------------------------------------------------------------------
# 10) Save the CasADi ODE function to file
# -----------------------------------------------------------------------------
casadi_file_path = os.path.join(save_dir, 'eom_2d_quadro_springpend_casadi.py')
with open(casadi_file_path, 'w') as f:
    f.write("import casadi as ca\n\n")
    f.write("def eom_2d_quadro_springpend(q, q_dot, q_ddot, w, w_dot, u, params):\n")
    f.write("    # params = [M, m, Ixx, g, k, l0, c, L, f_ext_dy, f_ext_dz, f_ext_my, f_ext_mz]\n")
    f.write("    M, m, Ixx, g, k, l0, c, L, f_ext_dy, f_ext_dz, f_ext_my, f_ext_mz = params\n")
    f.write("    # q:       [y, z, phi, r, theta]\n")
    f.write("    # q_dot:   [y_dot, z_dot, phi_dot, r_dot, theta_dot]\n")
    f.write("    # w1, w2:  rotor spin rates\n")
    f.write("    # u:       [u1, u2] = [dw1/dt, dw2/dt]\n")
    f.write("    # Return an array of eom[i] = 0\n")
    f.write("    eom = ca.vertcat(\n")
    for eq_str in eom_str_list:
        f.write(f"        {eq_str},\n")
    f.write("    )\n")
    f.write("    return eom\n")

print(f"CasADi-compatible ODE function saved to: {casadi_file_path}")
