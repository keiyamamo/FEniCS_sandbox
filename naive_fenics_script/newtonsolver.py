import matplotlib.pyplot as plt
from dolfin import *


class Problem(NonlinearProblem):
    def __init__(self, J, F, bcs):
        self.bilinear_form = J
        self.linear_form = F
        self.bcs = bcs
        NonlinearProblem.__init__(self)

    def F(self, b, x):
        assemble(self.linear_form, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        assemble(self.bilinear_form, tensor=A)
        for bc in self.bcs:
            bc.apply(A)


class CustomSolver(NewtonSolver):
    def __init__(self):
        NewtonSolver.__init__(self, mesh.mpi_comm(),
                              PETScKrylovSolver(), PETScFactory.instance())

    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)

        PETScOptions.set("ksp_type", "gmres")
        PETScOptions.set("ksp_monitor")
        PETScOptions.set("pc_type", "ilu")

        self.linear_solver().set_from_options()


mesh = UnitSquareMesh(32, 32)

V = FunctionSpace(mesh, "CG", 1)
g = Constant(1.0)
bcs = [DirichletBC(V, g, "near(x[0], 1.0) and on_boundary")]
u = Function(V)
v = TestFunction(V)
f = Expression("x[0]*sin(x[1])", degree=2)
F = inner((1 + u**2)*grad(u), grad(v))*dx - f*v*dx
J = derivative(F, u)

problem = Problem(J, F, bcs)
custom_solver = CustomSolver()
from IPython import embed; embed(); exit(1)
custom_solver.solve(problem, u.vector())

plt.figure()
plot(u, title="Solution")

plt.figure()
plot(grad(u), title="Solution gradient")

plt.show()