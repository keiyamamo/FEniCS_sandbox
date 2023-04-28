import mshr
import numpy as np

from .MovingCommon import mesh_velocity_setup, get_visualization_files, mesh_velocity_solve
from ..NSfracStep import *


# Override some problem specific parameters
def problem_parameters(NS_parameters, NS_expressions, commandline_kwargs, **NS_namespace):
    L = 1.0
    T = 1.0
    T_G = 4 * T
    # T = 4.0  # For err analysis, temporal
    A = 0.08  # Amplitude
    try:
        unstructured = commandline_kwargs["umesh"]
    except:
        unstructured = False

    NS_parameters.update(
        unstructured=unstructured,
        nu=0.025,
        T=T,
        A0=A,
        T_G=T_G,
        L=L,
        dt=0.005,
        Nx=20, Ny=20,
        folder="results_moving_vortex",
        plot_interval=1000,
        save_step=10000,
        checkpoint=10000,
        print_intermediate_info=10000,
        compute_error=1,
        use_krylov_solvers=True,
        velocity_degree=1,
        pressure_degree=1,
        max_iter=1,
        krylov_report=False)

    NS_parameters['krylov_solvers'] = {'monitor_convergence': False,
                                       'report': False,
                                       'relative_tolerance': 1e-12,
                                       'absolute_tolerance': 1e-12}
    NS_expressions.update(dict(
        initial_fields=dict(
            u0='-sin(2*pi*x[1])*exp(-4*pi*pi*nu*t)',
            u1=' sin(2*pi*x[0])*exp(-4*pi*pi*nu*t)',
            p=' -cos(2*pi*x[0])*cos(2*pi*x[1])*exp(-8*pi*pi*nu*t)'),
        initial_fields_w=dict(
            w0='A*2*pi / T_G * cos(2*pi*t/T_G)*sin(2*pi*(x[1] + L/2)/L)',
            w1='A*2*pi / T_G * cos(2*pi*t/T_G)*sin(2*pi*(x[0] + L/2)/L)'),
        total_error=np.zeros(3)))


def mesh(Nx, Ny, L, unstructured, dt, **params):
    if unstructured:
        domain = mshr.Rectangle(Point(-L / 2, -L / 2), Point(L / 2, L / 2))
        mesh = mshr.generate_mesh(domain, Nx)

    else:
        mesh = RectangleMesh(Point(-L / 2, -L / 2), Point(L / 2, L / 2), Nx, Ny)

    print("Mesh info: N_cells = {} |  dx={} | dt = {}".format(mesh.num_cells(), mesh.hmin(), dt))
    return mesh


def pre_boundary_condition(mesh, L, **NS_namespace):
    # Mark geometry
    boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary.set_all(0)

    inlet = AutoSubDomain(lambda x, b: b)
    inlet.mark(boundary, 1)

    return dict(boundary=boundary)


def create_bcs(V, Q, t, dt, VV, nu, sys_comp, boundary, initial_fields, **NS_namespace):
    for i, ui in enumerate(sys_comp):
        if 'IPCS' in NS_parameters['solver']:
            deltat = dt / 2. if ui == 'p' else 0.
        else:
            deltat = 0.
        ue = Expression((initial_fields[ui]),
                        element=VV[ui].ufl_element(),
                        t=t - deltat, nu=nu)
        NS_expressions["bc_%s" % ui] = ue

    bcs = dict((ui, []) for ui in sys_comp)
    bc0 = DirichletBC(V, NS_expressions["bc_u0"], boundary, 1)
    bc1 = DirichletBC(V, NS_expressions["bc_u1"], boundary, 1)
    bcp = DirichletBC(Q, NS_expressions["bc_p"], boundary, 1)

    bcs['u0'] = [bc0]
    bcs['u1'] = [bc1]
    bcs['p'] = [bcp]

    return bcs


def initialize(q_, q_1, q_2, VV, t, nu, dt, initial_fields, **NS_namespace):
    """Initialize solution.

    Use t=dt/2 for pressure since pressure is computed in between timesteps.

    """
    for ui in q_:
        if 'IPCS' in NS_parameters['solver']:
            deltat = dt / 2. if ui == 'p' else 0.
        else:
            deltat = 0.
        vv = interpolate(Expression((initial_fields[ui]),
                                    element=VV[ui].ufl_element(),
                                    t=t + deltat, nu=nu), VV[ui])
        q_[ui].vector()[:] = vv.vector()[:]
        if not ui == 'p':
            q_1[ui].vector()[:] = vv.vector()[:]
            deltat = -dt
            vv = interpolate(Expression((initial_fields[ui]),
                                        element=VV[ui].ufl_element(),
                                        t=t + deltat, nu=nu), VV[ui])
            q_2[ui].vector()[:] = vv.vector()[:]
    q_1['p'].vector()[:] = q_['p'].vector()[:]


def pre_solve_hook(V, mesh, t, nu, L, T_G, A0, newfolder, initial_fields_w, velocity_degree, x_, u_components, boundary,
                   **NS_namespace):
    # Visualization files
    viz_p, viz_u = get_visualization_files(newfolder)

    # Mesh velocity solver setup
    L_mesh, a_mesh, coordinates, dof_map, mesh_sol, u_vec \
        = mesh_velocity_setup(V, mesh, u_components, velocity_degree, x_)

    # Mesh velocity conditions
    w0 = Expression((initial_fields_w["w0"]), element=V.ufl_element(), t=t, nu=nu, L=L, T_G=T_G, A=A0)
    w1 = Expression((initial_fields_w["w1"]), element=V.ufl_element(), t=t, nu=nu, L=L, T_G=T_G, A=A0)
    NS_expressions["bc_w0"] = w0
    NS_expressions["bc_w1"] = w1

    bc_mesh = dict((ui, []) for ui in u_components)
    bc0 = DirichletBC(V, w0, boundary, 1)
    bc1 = DirichletBC(V, w1, boundary, 1)

    bc_mesh["u0"] = [bc0]
    bc_mesh["u1"] = [bc1]

    return dict(viz_p=viz_p, viz_u=viz_u, u_vec=u_vec, mesh_sol=mesh_sol, bc_mesh=bc_mesh, dof_map=dof_map,
                a_mesh=a_mesh, L_mesh=L_mesh, coordinates=coordinates)


def update_prescribed_motion(V, t, dt, wx_, u_components, coordinates, dof_map, w_, mesh_sol, bc_mesh, NS_expressions,
                             A_cache, a_mesh, L_mesh, **NS_namespace):
    for key, value in NS_expressions.items():
        if "bc" in key:
            if 'IPCS' in NS_parameters['solver'] and 'p' in key:
                deltat_ = dt / 2.
            else:
                deltat_ = 0.
            value.t = t - deltat_

    move = mesh_velocity_solve(A_cache, L_mesh, a_mesh, bc_mesh, coordinates, dof_map, dt, mesh_sol, u_components, w_,
                               wx_)
    return move


def temporal_hook(q_, t, nu, VV, dt, u_vec, p_, viz_u, viz_p, initial_fields, tstep, sys_comp,
                  compute_error, total_error, **NS_namespace):
    """Function called at end of timestep.

    Plot solution and compute error by comparing to analytical solution.
    Remember pressure is computed in between timesteps.

    """
    # Save solution
    assign(u_vec.sub(0), q_["u0"])
    assign(u_vec.sub(1), q_["u1"])

    viz_u.write(u_vec, t)
    viz_p.write(p_, t)

    if tstep % compute_error == 0:
        err = {}
        for i, ui in enumerate(sys_comp):
            if 'IPCS' in NS_parameters['solver']:
                deltat_ = dt / 2. if ui == 'p' else 0.
            else:
                deltat_ = 0.
            ue = Expression((initial_fields[ui]),
                            element=VV[ui].ufl_element(),
                            t=t - deltat_, nu=nu)
            ue = interpolate(ue, VV[ui])
            uen = norm(ue.vector())
            ue.vector().axpy(-1, q_[ui].vector())
            error = norm(ue.vector()) / uen
            err[ui] = "{0:2.6e}".format(norm(ue.vector()) / uen)
            total_error[i] += error * dt


def theend_hook(newfolder, mesh, q_, t, dt, nu, VV, sys_comp, total_error, initial_fields, **NS_namespace):
    final_error = np.zeros(len(sys_comp))
    for i, ui in enumerate(sys_comp):
        if 'IPCS' in NS_parameters['solver'] and ui == "p":
            deltat = dt / 2.
        else:
            deltat = 0.
        ue = Expression((initial_fields[ui]),
                        element=VV[ui].ufl_element(),
                        t=t - deltat, nu=nu)
        ue = interpolate(ue, VV[ui])
        final_error[i] = errornorm(q_[ui], ue)

    hmin = mesh.hmin()
    if MPI.rank(MPI.comm_world) == 0:
        print("hmin = {}".format(hmin))
    s0 = "Total Error:"
    s1 = "Final Error:"
    for i, ui in enumerate(sys_comp):
        s0 += " {0:}={1:2.6e}".format(ui, total_error[i])
        s1 += " {0:}={1:2.6e}".format(ui, final_error[i])

    if MPI.rank(MPI.comm_world) == 0:
        print(s0)
        print(s1)

    err_u = final_error[0]
    err_ux = final_error[0]
    err_uy = final_error[1]
    err_p = final_error[2]
    err_p_h = final_error[2]
    # Write errors to file
    error_array = np.asarray([round(t, 4), err_u, err_ux, err_uy, err_p, err_p_h])
    error_path = path.join(newfolder, "error_mms.txt")
    with open(error_path, 'a') as filename:
        filename.write(
            f"{error_array[0]} {error_array[1]} {error_array[2]} {error_array[3]} {error_array[4]} {error_array[5]}\n")
