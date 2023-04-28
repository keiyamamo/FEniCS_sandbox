from dolfin import *

mesh = UnitIntervalMesh(20)

# P1 = VectorElement("CG", mesh.ufl_cell(), 1)
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
TH = P1 * P1
M = FunctionSpace(mesh, TH)
V = FunctionSpace(mesh, P1)

# Some function in M to illustrate that components
# will change by assign
m = interpolate(Expression('x[0]', '1-x[0]',degree=2), M)
m0, m1 = split(m)

# Functions for components
v0 = interpolate(Expression('cos(pi*x[0])', degree=2), V)
v1 = interpolate(Expression('sin(pi*x[0])', degree=2), V)

from IPython import embed; embed(); exit(1)
