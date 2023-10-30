from dolfin import *

parameters["ghost_mode"] = "shared_facet"


assert MPI.size(MPI.comm_world) != 1 # Parallel only
# Generate mesh and save to file 
mesh = UnitSquareMesh(10, 10)
with HDF5File(mesh.mpi_comm(), 'mesh.h5', 'w') as f:
    f.write(mesh, 'mesh')

# Read mesh from file
with HDF5File(mesh.mpi_comm(), 'mesh.h5', 'r') as f:
    mesh = Mesh()
    f.read(mesh, 'mesh', True)

