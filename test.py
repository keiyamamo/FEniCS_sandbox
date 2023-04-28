from fenics import *
import numpy as np


def define_boundaries(mesh, space_dim):

    class Boundary_left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.0)

    class Boundary_right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 1.0)

    class Boundary_bottom(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 0.0)

    class Boundary_top(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 1.0)


    # Initialize sub-domain instances
    boundary_left = Boundary_left()
    boundary_right = Boundary_right()
    boundary_bottom = Boundary_bottom()
    boundary_top = Boundary_top()


    # Initialize mesh function for boundary domains
    boundaries = MeshFunction("size_t", mesh, space_dim - 1)
    boundaries.set_all(0)
    boundary_left.mark(boundaries, 1)
    boundary_right.mark(boundaries, 2)
    boundary_bottom.mark(boundaries, 3)
    boundary_top.mark(boundaries, 4)

    # Define new measures associated with the exterior boundaries
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)


    return ds

# Create mesh and define function space
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, 'CG', 2)
space_dim = mesh.geometry().dim()
ds = define_boundaries(mesh, space_dim)


# Define boundary condition
u_e = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)


# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)

r_l = 1000.0
s_l = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

r_r = -2.0
s_r = Expression('x[0]*x[0] + 2*x[1]*x[1]', degree=2)

r_b = 1000.0
s_b = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

r_t = -4.0
s_t = Expression('x[0]*x[0] + 2*x[1]*x[1]', degree=2)

F = dot(grad(u), grad(v))*dx \
  + r_l*(u-s_l)*ds(1) + r_r*(u-s_r)*ds(2) + r_b*(u-s_b)*ds(3) + r_b*(u-s_b)*ds(4) \
  - f*v*dx

a, L = lhs(F), rhs(F)


# Compute solution
u = Function(V)
solve(a == L, u)


# Compute error in L2 norm
error_L2 = errornorm(u_e, u, 'L2')

# Compute maximum error at vertices
vertex_values_u_D = u_e.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print('error_L2  =', error_L2)
print('error_max =', error_max)