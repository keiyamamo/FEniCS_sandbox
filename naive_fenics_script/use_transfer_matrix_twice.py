from dolfin import *
import numpy as np


def epsilon(u):
    """
    Computes the strain-rate tensor
    Args:
        u (Function): Velocity field

    Returns:
        epsilon (Function): Strain rate tensor of u
    """

    return 0.5 * (grad(u) + grad(u).T)


class STRESS:
    def __init__(self, u, p, nu, mesh):
        boundary_ds = Measure("ds", domain=mesh)
        boundary_mesh = BoundaryMesh(mesh, 'exterior')
        self.bmV = VectorFunctionSpace(boundary_mesh, 'CG', 1)

        # Compute stress tensor
        sigma = (2 * nu * epsilon(u)) - (p * Identity(len(u)))
        # Compute stress on surface
        n = FacetNormal(mesh)
        F = -(sigma * n)

        # Compute normal and tangential components
        Fn = inner(F, n)  # scalar-valued
        Ft = F - (Fn * n)  # vector-valued

        # Integrate against piecewise constants on the boundary
        scalar = FunctionSpace(mesh, 'DG', 0)
        vector = VectorFunctionSpace(mesh, 'CG', 1)
        scaling = FacetArea(mesh)  # Normalise the computed stress relative to the size of the element

        v = TestFunction(scalar)
        w = TestFunction(vector)

        # Create functions
        self.Fn = Function(scalar)
        self.Ftv = Function(vector)
        self.Ft = Function(scalar)

        self.Ln = 1 / scaling * v * Fn * boundary_ds
        self.Ltv = 1 / (2 * scaling) * inner(w, Ft) * boundary_ds
        self.Lt = 1 / scaling * inner(v, self.norm_l2(self.Ftv)) * boundary_ds

    def __call__(self):
        """
        Compute stress for given velocity field u and pressure field p

        Returns:
            Ftv_mb (Function): Shear stress
        """

        # Assemble vectors
        assemble(self.Ltv, tensor=self.Ftv.vector())
        self.Ftv_bm = interpolate(self.Ftv, self.bmV)

        return self.Ftv_bm

    def norm_l2(self, u):
        """
        Compute norm of vector u in expression form
        Args:
            u (Function): Function to compute norm of

        Returns:
            norm (Power): Norm as expression
        """
        return pow(inner(u, u), 0.5)


mesh = UnitCubeMesh(10, 10, 10)
refined_mesh = refine(mesh)

V2 = VectorFunctionSpace(mesh, "CG", 2)
V1 = VectorFunctionSpace(refined_mesh, "CG", 1)


f = Expression(("sin(x[0]*pi)", "cos(x[1]*pi)", "x[2]"), degree=2)

u_2 = interpolate(f, V2)

transfer_matrix = PETScDMCollection.create_transfer_matrix(V2, V1)

u_1 = Function(V1)
u_1.vector()[:] = transfer_matrix * u_2.vector()

stress = STRESS(u_1, 0.0, 1.0, refined_mesh)
tau = stress()

File("tau.pvd") << tau

back_transfer_matrix = PETScDMCollection.create_transfer_matrix(V1, V2)

u_2_back = Function(V2)
u_2_back.vector()[:] = back_transfer_matrix * u_1.vector()

stress_2 = STRESS(u_2_back, 0.0, 1.0, mesh)
tau_2 = stress_2()

File("tau_2.pvd") << tau_2