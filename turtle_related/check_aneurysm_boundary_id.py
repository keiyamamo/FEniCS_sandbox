from dolfin import *

"""
This script is meant to check the boundary id for aneurysm case
"""

# parameters
mesh_file = "final_mesh_100k"
fsi_id = 22
rigid_id = 11
outer_id = 33


mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), mesh_file + ".h5", "r")
hdf.read(mesh, "/mesh", False)
boundaries = MeshFunction("size_t", mesh, 2)
hdf.read(boundaries, "/boundaries")
domains = MeshFunction("size_t", mesh, 3)
hdf.read(domains, "/domains")

# Only considere FSI in domain within this sphere BC1
sph_x = 0.024
sph_y = 0.02
sph_z = 0.03
sph_rad = 0.008

i = 0
for submesh_facet in facets(mesh):
    idx_facet = boundaries.array()[i]
    if idx_facet == fsi_id:
        vert = submesh_facet.entities(0)
        mid = submesh_facet.midpoint()
        dist_sph_center = sqrt((mid.x()-sph_x)**2 + (mid.y()-sph_y)**2 + (mid.z()-sph_z)**2)
        if dist_sph_center > sph_rad:
            boundaries.array()[i] = rigid_id  # changed "fsi" idx to "rigid wall" idx
    if idx_facet == outer_id:
        vert = submesh_facet.entities(0)
        mid = submesh_facet.midpoint()
        dist_sph_center = sqrt((mid.x()-sph_x)**2 + (mid.y()-sph_y)**2 + (mid.z()-sph_z)**2)
        if dist_sph_center > sph_rad:
            boundaries.array()[i] = rigid_id  # changed "fsi" idx to "rigid wall" idx
    i += 1

# # Checking boundaries and domains
f = File('toto.pvd')
f << boundaries
f << domains
