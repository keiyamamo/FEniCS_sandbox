from dolfin import *
import numpy as np

mesh = UnitCubeMesh(10, 10, 5)
marker = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 1)

class wall(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

marker.set_all(0)
wall().mark(marker, 1)

submesh = BoundaryMesh(mesh, "exterior")
expr = Expression("x[0] + x[1]", degree=1)
V1 = FunctionSpace(mesh, "DG", 1)
u = interpolate(expr, V1)

V_sub = FunctionSpace(submesh, "DG", 1)
u_sub = interpolate(expr, V_sub)

# Create functin on sub mesh
v_sub = Function(V_sub)

# Get mapping from sub mesh cell to parent facet
sub_map = submesh.entity_map(mesh.topology().dim()-1).array()
sub_dofmap = V_sub.dofmap()

# Get copies of u vector and v vector (to assign afterwards)
u_vec = u.vector().get_local()
v_sub_copy = v_sub.vector().get_local()

mesh.init(mesh.topology().dim()-1, mesh.topology().dim())
f_to_c = mesh.topology()(mesh.topology().dim()-1, mesh.topology().dim())

dof_coords = V1.tabulate_dof_coordinates()
sub_coords = V_sub.tabulate_dof_coordinates()

for i, facet in enumerate(sub_map):
    cells = f_to_c(facet)
    # Get closure dofs on parent facet

    sub_dofs = sub_dofmap.cell_dofs(i)
    closure_dofs = V1.dofmap().entity_closure_dofs(
        mesh, mesh.topology().dim(), [cells[0]])
    copy_dofs =np.empty(len(sub_dofs), dtype=np.int32)

    for dof in closure_dofs:
        for j, sub_coord in enumerate(sub_coords[sub_dofs]):
            if np.allclose(dof_coords[dof], sub_coord):
                copy_dofs[j] = dof
                break
    sub_dofs = sub_dofmap.cell_dofs(i)
    # Copy data
    v_sub_copy[sub_dofs] = u_vec[copy_dofs]

v_sub.vector().set_local(v_sub_copy)

with XDMFFile("u.xdmf") as xdmf:
    xdmf.write_checkpoint(u, "u", 0.0, append=False)

with XDMFFile("V_sub.xdmf") as xdmf:
    xdmf.write_checkpoint(v_sub, "v", 0.0, append=False)