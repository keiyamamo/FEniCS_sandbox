from dolfin import *
import sys

linear_solver = sys.argv[1]
# Create mesh and define function space
mesh = UnitSquareMesh(100, 100)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
g = Expression("sin(5*x[0])", degree=2)
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# Compute solution
solver = LUSolver(Matrix(), linear_solver)
u = Function(V)
A = assemble(a)
b = assemble(L)
bc.apply(A, b)
solver.set_operator(A)
solver.solve(A, u.vector(), b)