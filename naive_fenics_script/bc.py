from dolfin import *

mesh = UnitSquareMesh(20, 20)
# Build function space
P2 = VectorElement("CG", mesh.ufl_cell(), 2)
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
TH = P2 * P1
VQ = FunctionSpace(mesh, TH)
bc0 = DirichletBC(VQ.sub(0), Expression(("1", "0"), degree=0), "std::abs(x[1])>1-1e-12 && on_boundary")
bc1 = DirichletBC(VQ.sub(0), Constant((0, 0)), "std::abs(x[0]*(1-x[0])*x[1])<1e-12 && on_boundary")