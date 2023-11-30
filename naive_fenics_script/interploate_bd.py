from IPython import embed
from dolfin import *
import numpy as np
mesh = UnitCubeMesh(10, 10, 5)
marker = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
for f in facets(mesh):
    marker[f] = (np.isclose(f.midpoint().x(), 0) or np.isclose(f.midpoint().x(), 1) or
                 np.isclose(f.midpoint().y(), 0) or np.isclose(f.midpoint().y(), 1))
submesh = create_meshview(marker, 1)

expr = Expression("x[0] + x[1]", degree=1)
V1 = FunctionSpace(mesh, "DG", 1)
u = interpolate(expr, V1)

print(
    f"Value of boundary integral is {assemble(u*ds(domain=mesh, subdomain_data=marker, subdomain_id=1))}")


V_sub = FunctionSpace(submesh, "DG", 1)
u_sub = interpolate(expr, V_sub)
print(f"SubMesh interpolated function {assemble(u_sub*dx(domain=submesh))}")

# Create functin on sub mesh
v_sub = Function(V_sub)

# Get mapping from sub mesh cell to parent facet
sub_map = submesh.topology().mapping()[0].cell_map()

tdim = mesh.topology().dim()
sub_dofmap = V_sub.dofmap()

# Get copies of u vector and v vector (to assign afterwards)
u_vec = u.vector().get_local()
sub_copy = v_sub.vector().get_local()
mesh.init(mesh.topology().dim()-1, mesh.topology().dim())
f_to_c = mesh.topology()(mesh.topology().dim()-1, mesh.topology().dim())

dof_coords = V1.tabulate_dof_coordinates()
for i, facet in enumerate(sub_map):
    f = Facet(mesh, facet)
    vertices = [Vertex(mesh, v_) for v_ in f.entities(0)]
    facet_coords = [v_.point().array() for v_ in vertices]
    cells = f_to_c(facet)
    # Get closure dofs on parent facet
    closure_dofs = V1.dofmap().entity_closure_dofs(
        mesh, mesh.topology().dim(), [cells[0]])
    copy_dofs = []
    for dof in closure_dofs:
        for f_coord in facet_coords:
            if np.allclose(dof_coords[dof], f_coord):
                copy_dofs.append(dof)
                break
    sub_dofs = sub_dofmap.cell_dofs(i)
    # Copy data
    sub_copy[sub_dofs] = u_vec[copy_dofs]
v_sub.vector().set_local(sub_copy)


print(assemble(v_sub*dx(domain=submesh)))

with XDMFFile("v_sub.xdmf") as xdmf:
    xdmf.write_checkpoint(v_sub, "v", 0.0, append=False)

with XDMFFile("u.xdmf") as xdmf:
    xdmf.write_checkpoint(u, "u", 0.0, append=False)