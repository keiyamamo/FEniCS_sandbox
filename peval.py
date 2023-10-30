from dolfin import *
import numpy as np

# Create mesh and define function space

mesh = UnitSquareMesh(2, 2)
# mesh = UnitCubeMesh(4, 4, 4)
V = FunctionSpace(mesh, "CG", 1)
u = interpolate(Expression("x[0]", degree=1), V)

# make sure that x is not at vertex
x = np.array([0.509, 0.509])

# evaluate u at x
try:
    u_x = u(x)
except RuntimeError:
    u_x = yloc = np.inf * np.ones(u.value_shape())

cell_index = mesh.bounding_box_tree().compute_first_collision(Point(x))
print(f"cell_index = {cell_index}")
print("u(x) = {}".format(u_x))