from dolfin import *
import numpy as np

mesh = UnitSquareMesh(10, 10)
mesh.init(1)
ALE.move(mesh, Expression(("x[0]", "cos(x[1])"),degree=1))
num_cells = mesh.num_cells()
V = FunctionSpace(mesh, "DG", 0)
u = Function(V)
values = np.zeros(num_cells, dtype=np.float64)

for cell in cells(mesh):
    edges = cell.entities(1)
    
    value = 0
    for edge in edges:
        value += Edge(mesh, edge).length()
    values[cell.index()] = value / len(edges)
u.vector().set_local(values)
u.vector().apply("local")
with XDMFFile("edge_length.xdmf") as xdmf:
    xdmf.write(u)