import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.linalg import toeplitz
from scipy.interpolate import interp1d
from sklearn.linear_model import OrthogonalMatchingPursuit


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'

n_elements = 1000
L = 10.0
Tfinal = 10.0
target_CFL = 1.0
mesh = fe.IntervalMesh(n_elements, 0.0, L)

mean_velocity = 30 # cm/s

# Define time stepping of implicit Euler method (=dt)
h = L / n_elements
dt = target_CFL * h / mean_velocity
n_steps = mean_velocity * 1000 # for this specific length and duration #int(np.ceil(Tfinal / dt))

# Define velocity and diffusion coefficient
#velocity = fe.Constant((50.0,))
diffusion = fe.Constant(0.04)

# Input Velocity (Unsteady)
heart_cycle = 1.0 # seconds = 60 BPM
num_points = mean_velocity * 100
t = np.linspace(0, heart_cycle, num_points, endpoint=True)

systole_end = 0.32   # systole ~32% of cycle

flow = np.full_like(t, 0.25 * mean_velocity)   # baseline

syst = t < systole_end
flow[syst] += 0.85 * mean_velocity * np.sin(np.pi * t[syst] / systole_end)**2

diast = t >= systole_end
diast_duration = heart_cycle - systole_end

x = np.pi * (t[diast] - systole_end) / diast_duration
k = 0.8
flow[diast] += mean_velocity * 1/np.arcsin(k)*np.arctan(k*np.sin(x)/(1 - k*np.cos(x)))


# Force exact mean 
flow = flow * (mean_velocity / np.mean(flow))


# Imaging duration: 10 heart cycle
final_velocity = np.tile(flow, 10) # reps = Tfinal / heart_cycle duration


A=200.0; Ts=0.0; alpha=2.0; b=2.0
t = np.linspace(0, Tfinal, n_steps)
inlet_contrast = np.zeros_like(t)

inlet_contrast += A * pow(t - Ts, alpha) / pow(b, alpha) * np.exp( -(t - Ts)/b ) * (t >= Ts)

plt.figure()
plt.plot(t, inlet_contrast, label='inlet contrast')
plt.plot(t, final_velocity, label='velocity (cm/s)')
plt.legend()
plt.show()



# Define function space
lagrange_polynomial_space_first_order = fe.FunctionSpace(
    mesh, "Lagrange", 1
)

u_D = fe.Constant(0.0)


inlet_expr_str = (
    "A * pow(t - Ts, alpha) / pow(b, alpha) * exp( -(t - Ts)/b ) * (t >= Ts)"
)

u_D = fe.Expression(inlet_expr_str, degree=2, A=1.0, Ts=0.0, alpha=2.0, b=2.0, t=0.0)


# Define boundary condition function to return whether we are on the boundary
def boundary_boolean_function(x, on_boundary):
    return on_boundary and fe.near(x[0], 0.0)

# The non-homogeneous Dirichlet boundary condition
boundary_condition = fe.DirichletBC(
    lagrange_polynomial_space_first_order,
    u_D,
    boundary_boolean_function,
)

# Define initial condition
u_old = fe.Function(lagrange_polynomial_space_first_order)
u_old.vector()[:] = 0.0

# The force on the right-hand side
f = fe.Constant(0.0)

# Create the Finite Element variational problem
u = fe.TrialFunction(lagrange_polynomial_space_first_order)
v = fe.TestFunction(lagrange_polynomial_space_first_order)


velocity = fe.Constant((final_velocity[0],))


# Weak form of the Advection-Diffusion equation
weak_form_residuum = (
    u * v * fe.dx
    +
    dt * fe.dot(velocity, fe.grad(u)) * v * fe.dx
    +
    dt * diffusion * fe.dot(fe.grad(u), fe.grad(v)) * fe.dx
    -
    (
        u_old * v * fe.dx
        +
        dt * f * v * fe.dx
    )
)

# Convert to linear system
weak_form_lhs = fe.lhs(weak_form_residuum)
weak_form_rhs = fe.rhs(weak_form_residuum)

# Prepare solution function
u_solution = fe.Function(
    lagrange_polynomial_space_first_order
)

# Time-stepping loop
t_current = 0.0

# Store final solution for plotting u_final(t,x)
u_final = np.zeros((n_steps + 1, n_elements + 1))
u_final[0, :] = u_old.vector().get_local()

for i in range(n_steps):
    t_current += dt
    u_D.t = t_current
    velocity.assign(fe.Constant((final_velocity[i],)))

    # Assemble system, BC applied here
    fe.solve(
        weak_form_lhs == weak_form_rhs,
        u_solution,
        boundary_condition,
    )

    # Update for next time step
    u_old.assign(u_solution)

    # Store solution
    u_final[i + 1, :] = u_solution.vector().get_local()

'''
# Plot results as an image
plt.figure(figsize=(8,6))
plt.imshow(
    u_final.T,                      
    extent=[0, n_steps*dt, 0, L],   
    origin='lower',
    aspect='auto'
)
plt.xlabel("time (s)")
plt.ylabel("x (cm)")
plt.title(f"Concentration u(x,t) over Space and Time\nPDE Simulation, velocity = {velocity.values()[0]} cm/s")
plt.colorbar(label="u(x,t)")
plt.show()
'''


outfile = "contrast.txt"
timefile = "time.txt"


data = np.column_stack(u_final)
np.savetxt(
    outfile,
    data,
    fmt="%.6f",
    comments="# "
)

np.savetxt(
    timefile, t, fmt="%0.6f"
)



# 3D plot of the results along t at specific x locations
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
plt.set_cmap('viridis')
x_locations = np.arange(0, L+0.1, 2.0)
x_indices = [int(x_loc / h) for x_loc in x_locations]
time_points = np.linspace(0, n_steps*dt, u_final.shape[0])
for idx in x_indices:
    ax.plot(time_points, u_final[:, idx], zs=idx*h, zdir='y', label=f'x={idx*h:.1f} cm')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position x (cm)')
ax.set_zlabel('Concentration u')
plt.title('Concentration vs Time at Different Positions')
plt.legend()
plt.show()
