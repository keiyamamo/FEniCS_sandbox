from dolfin import *
import numpy as np
mesh = UnitSquareMesh(10, 10)
V = VectorFunctionSpace(mesh, "CG", 1)
u = Function(V)
u_ = Function(V)
u.interpolate(Expression(("3*x[0]", "2*x[1]"), degree=1))
from IPython import embed; embed(); exit(1)
num_dofs_per_component = int(V.dim()/V.num_sub_spaces())
num_sub_spaces = V.num_sub_spaces()
print(num_dofs_per_component, num_sub_spaces)
vector = np.zeros((num_sub_spaces, num_dofs_per_component))
for i in range(num_sub_spaces):
    vector[i] = u.sub(i, deepcopy=True).vector().get_local()
x = V.sub(0).collapse().tabulate_dof_coordinates()
vector = vector.T
for coord, vec in zip(x, vector):
    print(coord, vec)