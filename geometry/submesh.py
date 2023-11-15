from dolfin import *

mesh = UnitSquareMesh(20, 20)


class left(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.2 and x[1] <= 0.5 + 1e3 * DOLFIN_EPS


class top(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.2 and x[1] > 0.5 - 1e3 * DOLFIN_EPS


mf = MeshFunction("size_t", mesh, mesh.topology().dim(), 1)

left().mark(mf, 2)
top().mark(mf, 3)


merged_mf = MeshFunction("size_t", mesh, mesh.topology().dim(), 1)
tmp_array = merged_mf.array().copy()
tmp_array[mf.where_equal(2)] = 5
tmp_array[mf.where_equal(3)] = 5
merged_mf.set_values(tmp_array)

solid_mesh = SubMesh(mesh, merged_mf, 5)


mf_sub = MeshFunction("size_t", solid_mesh, solid_mesh.topology().dim(), 1)
parent_cell_indices = solid_mesh.data().array(
    "parent_cell_indices", solid_mesh.topology().dim())
tmp_sub_array = mf_sub.array().copy()
tmp_sub_array[:] = mf.array()[parent_cell_indices]
print(tmp_sub_array)
mf_sub.set_values(tmp_sub_array)

File("SubMesh_mf.pvd") << mf_sub

File("MF.pvd") << mf