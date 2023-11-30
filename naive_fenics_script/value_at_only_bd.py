from fenics import *

mesh = UnitSquareMesh(8, 8)

Right = AutoSubDomain(lambda x, on_bnd: near(x[0], 1) and on_bnd)

V = FunctionSpace(mesh, "CG", 1)

bc = DirichletBC(V, Constant((10)), Right)

u = Function(V)

bc.apply(u.vector())

u_divide = Function(V)

boundary_dof = bc.get_boundary_values().keys()
from IPython import embed; embed(); exit(1)
u_divide.vector()[boundary_dof] = 1.0 / (u.vector()[boundary_dof])



