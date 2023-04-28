from dolfin import *
import numpy as np
parameters["ghost_mode"] = "shared_facet"

L = 0.2
H = 0.2
if MPI.comm_world.rank == 0:
    mesh = UnitSquareMesh(MPI.comm_self, 10, 10)

    class Domain(SubDomain):
        def inside(self, x, on_boundary):
            return abs(x[0] - 0.5) < L/2 + 100*DOLFIN_EPS and abs(x[1] - 0.5) < H/2 + 100*DOLFIN_EPS
    tdim = mesh.topology().dim()
    # Mark part of the domain
    cf = MeshFunction('size_t', mesh, mesh.topology().dim(), 1)

    Domain().mark(cf, 2)

    # Find interface between domains
    ff = MeshFunction("size_t", mesh, tdim-1, 1)
    mesh.init(tdim-1, tdim)
    f_to_c = mesh.topology()(tdim-1, tdim)
    for facet in facets(mesh):
        cells = f_to_c(facet.index())
        values = cf.array()[cells]
        if len(np.unique(values)) == 2:
            ff.array()[facet.index()] = 2
    with XDMFFile(MPI.comm_self, "ff.xdmf") as xdmf:
        xdmf.write(ff)
    with XDMFFile(MPI.comm_self, "cf.xdmf") as xdmf:
        xdmf.write(mesh)
        xdmf.write(cf)
MPI.comm_world.Barrier()
mesh = Mesh()
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile("cf.xdmf") as xdmf:
    xdmf.read(mesh)
    xdmf.read(mvc)
cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
with XDMFFile("ff.xdmf") as xdmf:
    xdmf.read(mvc)
ff = cpp.mesh.MeshFunctionSizet(mesh, mvc)

# Intgral over a s

dS = Measure("dS", domain=mesh, subdomain_data=ff)
dx = Measure("dx", domain=mesh, subdomain_data=cf)
from IPython import embed; embed(); exit(1)
V = FunctionSpace(mesh, "DG", 0)
u = TrialFunction(V)
v = TestFunction(V)
ah = inner(u, v)*dx
Lh = inner(3, v)*dx(2) + inner(7, v)*dx(1)
uh = Function(V)
solve(ah == Lh, uh)

n = FacetNormal(mesh)
x = SpatialCoordinate(mesh)
integral = uh("+")*dS(2)
restriction = Constant(0)*dx
print("No restriction:", assemble(integral))
print("Restriction:", assemble(integral+restriction))
print("Exact:", 2*3*L + 2*3*H)