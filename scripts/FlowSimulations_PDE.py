import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

if __name__ == "__main__":
    n_elements = 1000
    L = 1.0
    Tfinal = 30.0
    target_CFL = 2.0
    mesh = fe.UnitIntervalMesh(n_elements)

    # Define velocity and diffusion coefficient
    velocity = fe.Constant((4.0,))
    diffusion = fe.Constant(0.004)

    # Define function space
    lagrange_polynomial_space_first_order = fe.FunctionSpace(
        mesh, "Lagrange", 1
    )

    # Define time-dependent boundary condition at x=0 using a Gaussian pulse
    A = 800.0
    t0 = 15.0
    sigma = 5.0

    u_D = fe.Constant(0.0)

    def inlet_value(t):
        return A*np.exp(-0.5*((t - t0)/sigma)**2)

    # Define boundary condition function to return whether we are on the boundary
    def boundary_boolean_function(x, on_boundary):
        return on_boundary and fe.near(x[0], 0.0)
    
    # The homogeneous Dirichlet boundary condition
    boundary_condition = fe.DirichletBC(
        lagrange_polynomial_space_first_order,
        u_D,
        boundary_boolean_function,
    )

    # Define initial condition
    u_old = fe.Function(lagrange_polynomial_space_first_order)
    u_old.vector()[:] = 0.0

    # Define time stepping of implicit Euler method (=dt)
    h = L / n_elements
    dt = target_CFL * h / velocity.values()[0]
    n_steps = int(np.ceil(Tfinal / dt))

    # The force on the right-hand side
    f = fe.Constant(0.0)

    # Create the Finite Element variational problem
    u = fe.TrialFunction(lagrange_polynomial_space_first_order)
    v = fe.TestFunction(lagrange_polynomial_space_first_order)


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

    # Store final solution for plotting
    u_final = np.zeros((n_steps + 1, n_elements + 1))
    u_final[0, :] = u_old.vector().get_local()

    for i in range(n_steps):
        t_current += dt
        u_D.assign(inlet_value(t_current))

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

    # Plot results as an image
    """
    plt.figure(figsize=(8,6))
    plt.imshow(
        u_final.T,                      # transpose so cols=time, rows=space
        extent=[0, n_steps*dt, 0, L],   # x: time, y: x
        origin='lower',
        aspect='auto'
    )
    plt.xlabel("t")
    plt.ylabel("x")
    plt.colorbar(label="u(x,t)")
    plt.show()
    """

    time_points = np.linspace(0, n_steps*dt, u_final.shape[0])
    ref = u_final[15,:]
    u_washout = u_final[time_points > t0, :]
    delays = np.zeros(u_washout.shape[0])
    for i in range(u_washout.shape[0]):
        corr = correlate(u_washout[i,:] - np.mean(u_washout[i,:]), ref - np.mean(ref), mode='full')
        delay_idx = np.argmax(corr) - (len(ref))
        delays[i] = delay_idx * h

    id_max = np.argmax(delays)
    time_points = np.linspace(0, n_steps*dt/2, u_washout.shape[0])
    m,b = np.polyfit(time_points[time_points<0.1], delays[time_points<0.1], 1)
    v_estimated = m
    print(f'Estimated velocity: {v_estimated:.2f} cm/s (True velocity: {velocity.values()[0]} cm/s)')

    plt.figure(figsize=(6,4))
    plt.plot(time_points, delays, label='Estimated delays')
    plt.plot(time_points[time_points<0.1], m*time_points[time_points<0.1] + b, 'r--', label=f'Fit: v={v_estimated:.2f} cm/s')
    plt.xlabel('time'); plt.ylabel('Delay (s)'); plt.title('Estimated Delays vs Position'); plt.legend()
    plt.show()

    print("time_difference:", dt)
        