from dolfin import *


def local_project(f, V):
    u = TrialFunction(V)
    v = TestFunction(V)
    a_proj = inner(u, v)*ds
    b_proj = inner(f, v)*ds
    A = assemble(a_proj, keep_diagonal=True)
    A.ident_zeros()
    b = assemble(b_proj)
    # solver = LocalSolver(A, b)
    # solver.factorize()
    u = Function(V)
    # solver.solve_local_rhs(u)
    solve(A, u.vector(), b)
    # from IPython import embed; embed(); exit(1)
    return u


class STRESS:
    def __init__(self, u, nu, mesh, velocity_degree):
        boundary_ds = Measure("ds", domain=mesh)
        boundary_mesh = BoundaryMesh(mesh, 'exterior')
        self.bmV = VectorFunctionSpace(boundary_mesh, 'DG', velocity_degree -1)
        self.V = VectorFunctionSpace(mesh, 'DG', velocity_degree -1)

        # Compute stress tensor
        sigma = (2 * nu * sym(grad(u)))
        # Compute stress on surface
        n = FacetNormal(mesh)
        F = -(sigma * n)

        # Compute normal and tangential components
        Fn = inner(F, n)  # scalar-valued
        self.Ft = F - (Fn * n)  # vector-valued

    def __call__(self):
        """
        Compute stress for given velocity field u

        Returns:
            Ftv_mb (Function): Shear stress
        """
        self.Ftv = local_project(self.Ft, self.V)
        self.Ftv_bd = interpolate(self.Ftv, self.bmV)

        return self.Ftv, self.Ftv_bd

velocity_degree = 2
mesh = UnitCubeMesh(10, 10, 10)

V = VectorFunctionSpace(mesh, "CG", velocity_degree)
f = Expression(("10*x[0]*x[0]+sin(x[0])", "cos(x[1])", "x[2]"), degree=velocity_degree)
u_2 = interpolate(f, V)
volume_writer = XDMFFile("u2.xdmf")
volume_writer.write_checkpoint(u_2, "u", 0, XDMFFile.Encoding.HDF5, False)
volume_writer.close()


stress = STRESS(u_2, 1, mesh, velocity_degree)
volume_tau, surface_tau = stress()


volume_writer = XDMFFile("volume_tau.xdmf")
volume_writer.write_checkpoint(volume_tau, "tau", 0, XDMFFile.Encoding.HDF5, False)
volume_writer.close()
surface_tau_writer = XDMFFile("surface_tau.xdmf")
surface_tau_writer.write_checkpoint(surface_tau, "tau", 0, XDMFFile.Encoding.HDF5, False)
surface_tau_writer.close()

