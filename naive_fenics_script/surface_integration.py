from fenics import UnitCubeMesh, VectorFunctionSpace, TestFunction, Function, \
    Constant, assemble, FacetArea, inner, ds, File

# Create a 3D cube mesh
meshsize = 10
mesh = UnitCubeMesh(meshsize, meshsize, meshsize)

# Define function spaces
V0 = VectorFunctionSpace(mesh, 'DG', 0)  # Discontinous Galerkin const.
VC1 = VectorFunctionSpace(mesh, 'CG', 1)  # Continous Galerkin 1-order

# Define the test functions
v0 = TestFunction(V0)
vc1 = TestFunction(VC1)

# Define functions to be integrated
u_0 = Constant((1., 0., 0.))
u_00 = Function(V0)

u_c1 = Constant((1., 0., 0.))
u_c11 = Function(VC1)


# Integrate the functions elementwise
assemble((1 / FacetArea(mesh)) * inner(v0, u_0) * ds, tensor=u_00.vector())
assemble((1 / FacetArea(mesh)) * inner(vc1, u_c1) * ds, tensor=u_c11.vector())

File('u_00.pvd') << u_00
File('u_c11.pvd') << u_c11
