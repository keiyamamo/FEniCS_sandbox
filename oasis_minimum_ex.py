from ..NSfracStep import *
from ..DrivenCavity import *

# Override some problem specific parameters
def problem_parameters(NS_parameters, **NS_namespace):
    NS_parameters.update(
        nu=0.001,
        T=1.0,
        dt=0.001,
        folder="drivencavity_results")

def mesh(Nx=50, Ny=50, **params):
    return UnitSquareMesh(Nx, Ny)
    
# Specify boundary conditions
def create_bcs(V, **NS_namespace):
    noslip = "std::abs(x[0]*x[1]*(1-x[0]))<1e-8"
    top = "std::abs(x[1]-1) < 1e-8"
    bc0 = DirichletBC(V, 0, noslip)
    bc00 = DirichletBC(V, 1, top)
    bc01 = DirichletBC(V, 0, top)
    return dict(u0=[bc00, bc0], u1=[bc01, bc0],p=[])

def initialize(x_1, x_2, bcs, **NS_namespace):
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]
