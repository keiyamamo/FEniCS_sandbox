import gmsh
import dolfinx
import dolfinx_mpc
from dolfinx.fem import FunctionSpace, Function, Constant
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.io.gmshio import model_to_mesh

from ufl import grad, dot, lhs, rhs, Measure, TrialFunction, TestFunction, FiniteElement
import ufl
from petsc4py import PETSc
import numpy as np
from mpi4py import MPI

# MPI initialization
comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.rank

##################################################
## Mesh generation Start
##################################################
gmsh.initialize()
r = 0.1     # radius of sphere
L = 1.0     # length of box

gdim = 3         # Geometric dimension of the mesh
fdim = gdim - 1  # facet dimension

if rank == 0:
    # Define geometry for RVE
    sphere = gmsh.model.occ.add_sphere(L/2, L/2, L/2, r)
    box = gmsh.model.occ.add_box(0.0, 0.0, 0.0, L, L, L)
    whole_domain = gmsh.model.occ.fragment([(3, box)], [(3, sphere)])
    gmsh.model.occ.synchronize()

    matrix_physical = gmsh.model.addPhysicalGroup(gdim, [box], tag=1)         # tag for matrix
    inclusion_physical = gmsh.model.addPhysicalGroup(gdim, [sphere], tag=2)      # tag for inclusion

    # Get the interface elements between rectangle and circle and tag
    interface_elements = gmsh.model.getBoundary([(gdim, inclusion_physical)])
    interface_physical = gmsh.model.addPhysicalGroup(fdim, (1, 1), tag = 9)

    background_surfaces = []
    inclusion_surfaces = []
    for domain in whole_domain[0]:
        com = gmsh.model.occ.getCenterOfMass(domain[0], domain[1])
        mass = gmsh.model.occ.getMass(domain[0], domain[1])
        if np.isclose(mass, 4/3 * np.pi * (r ** 3)):  # identify inclusion by its mass
            inclusion_surfaces.append(domain)
        else:
            background_surfaces.append(domain)

    gmsh.model.mesh.field.add("Distance", 1)
    edges = gmsh.model.getBoundary(inclusion_surfaces, oriented=False)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [e[0] for e in edges])
    gmsh.model.mesh.field.setNumber(1, "Sampling", 300)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", 0.010)
    gmsh.model.mesh.field.setNumber(2, "LcMax", 0.030)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.00)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 0.075)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(gdim)

domain, ct, _ = model_to_mesh(gmsh.model, comm, rank, gdim=gdim)
gmsh.finalize()

dx = Measure('dx', domain=domain, subdomain_data=ct)
x = ufl.SpatialCoordinate(domain)
##################################################
## Mesh generation finished
##################################################

# elements and funcionspace
###########################

P1 = FiniteElement("CG", domain.ufl_cell(), 1)
V = FunctionSpace(domain, P1)

# define function, trial function and test function
T_fluc = TrialFunction(V)
dT_fluc = TestFunction(V)

# define material parameter
S = FunctionSpace(domain, ("DG", 0))
kappa = Function(S)
matrix = ct.find(1)
kappa.x.array[matrix] = np.full_like(matrix, 5, dtype=PETSc.ScalarType)
inclusion = ct.find(2)
kappa.x.array[inclusion]  = np.full_like(inclusion, 1, dtype=PETSc.ScalarType)
kappa.x.scatter_forward()

# boundary conditions
#####################
bcs = []
## periodic boundary conditions
pbc_directions = []
pbc_slave_tags = []
pbc_is_slave = []
pbc_is_master = []
pbc_meshtags = []
pbc_slave_to_master_maps = []

mpc = dolfinx_mpc.MultiPointConstraint(V)

def generate_pbc_slave_to_master_map(i):
    def pbc_slave_to_master_map(x):
        out_x = x.copy()
        out_x[i] = x[i] - L
        return out_x

    return pbc_slave_to_master_map

def generate_pbc_is_slave(i):
    return lambda x: np.isclose(x[i], L)

def generate_pbc_is_master(i):
    return lambda x: np.isclose(x[i], 0.0)

for i in range(gdim):

    pbc_directions.append(i)
    pbc_slave_tags.append(i + 3)
    pbc_is_slave.append(generate_pbc_is_slave(i))
    pbc_is_master.append(generate_pbc_is_master(i))
    pbc_slave_to_master_maps.append(generate_pbc_slave_to_master_map(i))

    facets = locate_entities_boundary(domain, fdim, pbc_is_slave[-1])
    arg_sort = np.argsort(facets)
    pbc_meshtags.append(meshtags(domain,
                                 fdim,
                                 facets[arg_sort],
                                 np.full(len(facets), pbc_slave_tags[-1], dtype=np.int32)))

N_pbc = len(pbc_directions)
for i in range(N_pbc):

    # slave/master mapping of opposite surfaces (without slave-slave[-slave] intersections)
    def pbc_slave_to_master_map(x):
        out_x = pbc_slave_to_master_maps[i](x)
        # remove edges that are connected to another slave surface
        idx = np.logical_or(pbc_is_slave[(i + 1) % N_pbc](x),pbc_is_slave[(i + 2) % N_pbc](x))
        out_x[pbc_directions[i]][idx] = np.nan
        return out_x

    mpc.create_periodic_constraint_topological(V, pbc_meshtags[i],
                                                   pbc_slave_tags[i],
                                                   pbc_slave_to_master_map,
                                                   bcs)

if len(pbc_directions) > 1:
    def pbc_slave_to_master_map_corner(x):
        '''
        Maps the slave corner dof [intersection of slave_x, slave_y, slave_z] (1,1,1) to
        master corner dof [master_x, master_y, master_z] (0,0,0)
        '''
        out_x = x.copy()
        out_x[0] = x[0] - L
        out_x[1] = x[1] - L
        out_x[2] = x[2] - L
        idx_corner = np.logical_and(pbc_is_slave[0](x), np.logical_and(pbc_is_slave[1](x), pbc_is_slave[2](x)))
        out_x[0][~idx_corner] = np.nan
        out_x[1][~idx_corner] = np.nan
        out_x[2][~idx_corner] = np.nan
        return out_x

    def generate_pbc_slave_to_master_map_edges(dir_i, dir_j):
        def pbc_slave_to_master_map_edges(x):
            '''
            Maps the slave edge dofs [intersection of slave_i, slave_j] (i=1,j=1) to
            master corner dof [master_i, master_j] (i=0,j=0)
            (1) map slave_x, slave_y to master_x, master_y: i=0, j=1
            (2) map slave_x, slave_z to master_x, master_z: i=0, j=2
            (3) map slave_y, slave_z to master_y, master_z: i=1, j=2
            '''
            out_x = x.copy()
            out_x[dir_i] = x[dir_i] - L
            out_x[dir_j] = x[dir_j] - L
            idx = np.logical_and(pbc_is_slave[dir_i](x), pbc_is_slave[dir_j](x))
            out_x[dir_i][~idx] = np.nan
            out_x[dir_j][~idx] = np.nan
            idx_corner = np.logical_and(pbc_is_slave[0](x), np.logical_and(pbc_is_slave[1](x), pbc_is_slave[2](x)))
            out_x[dir_i][idx_corner] = np.nan
            out_x[dir_j][idx_corner] = np.nan
            print(len(out_x[dir_i][idx]))
            return out_x
        return pbc_slave_to_master_map_edges

    mapping_slave_intersections = [(0,1),(0,2),(1,2)] # pairs of slave intersections

    for ij in range(gdim):
        # mapping slave intersection node (corner) to master intersection node (opposite corner)
        mpc.create_periodic_constraint_topological(V, pbc_meshtags[0],
                                                   pbc_slave_tags[0],
                                                   pbc_slave_to_master_map_corner,
                                                   bcs)

        for inters in mapping_slave_intersections:
            # mapping slave intersections to opposite master intersections
            mpc.create_periodic_constraint_topological(V, pbc_meshtags[inters[0]],
                                                       pbc_slave_tags[inters[0]],
                                                       generate_pbc_slave_to_master_map_edges(inters[0], inters[1]),
                                                       bcs)

mpc.finalize()

# Variational problem
#####################
T_gradient = Constant(domain, PETSc.ScalarType((1,0,0)))

## weak formulation
eqn = - kappa * dot(grad(dot(T_gradient , x) + T_fluc), grad(dT_fluc)) * dx
a_form = lhs(eqn)
L_form = rhs(eqn)

## solving
func_sol = Function(V)
problem = dolfinx_mpc.LinearProblem(a_form, L_form, mpc, bcs=bcs)
func_sol = problem.solve()

# write results to file
file_results1 = XDMFFile(domain.comm, "results_3D_MWE/T_fluc.xdmf", "w")
file_results1.write_mesh(domain)
file_results1.write_function(func_sol)
file_results1.close()