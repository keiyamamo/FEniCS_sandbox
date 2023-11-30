from dolfin import *
import numpy as np

mesh = UnitCubeMesh(10, 10, 5)
marker = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 1)

class wall(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

marker.set_all(0)
wall().mark(marker, 1)

submesh_boundary_version = BoundaryMesh(mesh, "exterior")
submesh = MeshView.create(marker, 1)
expr = Expression("x[0] + x[1]", degree=1)
V1 = FunctionSpace(mesh, "DG", 1)
u = interpolate(expr, V1)

print(
    f"Value of boundary integral is {assemble(u*ds(domain=mesh, subdomain_data=marker, subdomain_id=1))}")

#NOTE: V_sub.tabulate_dof_coordinates() and V_sub_boundary.tabulate_dof_coordinates() are NOT the same
V_sub = FunctionSpace(submesh, "DG", 1)
V_sub_boundary = FunctionSpace(submesh_boundary_version, "DG", 1)


u_sub = interpolate(expr, V_sub)
u_sub_boundary = interpolate(expr, V_sub_boundary)
print(f"SubMesh interpolated function {assemble(u_sub*dx(domain=submesh))}")
print(f"SubMeshBoundary interpolated function {assemble(u_sub_boundary*dx(domain=submesh_boundary_version))}")

# Create functin on sub mesh
v_sub = Function(V_sub)
v_sub_boundary = Function(V_sub_boundary)

# Get mapping from sub mesh cell to parent facet
sub_map = submesh.topology().mapping()[0].cell_map()
sub_map_boundary_version = submesh_boundary_version.entity_map(mesh.topology().dim()-1).array()
# Check that both are the same
assert np.allclose(sub_map, sub_map_boundary_version)

sub_dofmap = V_sub.dofmap()
sub_dofmap_boundary_version = V_sub_boundary.dofmap()

# Get copies of u vector and v vector (to assign afterwards)
u_vec = u.vector().get_local()
u_vec_copy = u.vector().get_local()
sub_copy = v_sub.vector().get_local()
v_sub_copy = v_sub_boundary.vector().get_local()

mesh.init(mesh.topology().dim()-1, mesh.topology().dim())
f_to_c = mesh.topology()(mesh.topology().dim()-1, mesh.topology().dim())

dof_coords = V1.tabulate_dof_coordinates()
sub_coords = V_sub_boundary.tabulate_dof_coordinates()
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
    sub_dofs_boundary_version = sub_dofmap_boundary_version.cell_dofs(i)
    assert np.allclose(sub_dofs, sub_dofs_boundary_version)
    # Copy data
    # sub_copy[sub_dofs] = u_vec[copy_dofs]
    v_sub_copy[sub_dofs_boundary_version] = u_vec[copy_dofs]

# v_sub.vector().set_local(sub_copy)
v_sub_boundary.vector().set_local(v_sub_copy)



print(assemble(v_sub*dx(domain=submesh)))

with XDMFFile("v_sub.xdmf") as xdmf:
    xdmf.write_checkpoint(v_sub, "v", 0.0, append=False)

with XDMFFile("u.xdmf") as xdmf:
    xdmf.write_checkpoint(u, "u", 0.0, append=False)

with XDMFFile("v_sub_boundary.xdmf") as xdmf:
    xdmf.write_checkpoint(v_sub_boundary, "v", 0.0, append=False)