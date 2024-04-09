from fenics import *
# constructing mesh on (0, 1)
N = 49
mesh = UnitIntervalMesh(N); V = FunctionSpace(mesh, 'P', 1)
# (step)-function of interest
u = interpolate(Expression('x[0] >= 0.3 && x[0] <= 0.7 ? 2 : 1', degree = 2), V)

# Function to project and plot:
f = sqrt(dot(grad(u),grad(u)))

# Option 0: Consistent-mass projection to a continous space is oscillatory when
# projecting discontinuous functions.
plot(project(f))

# Option 1: Project to a DG0 space (i.e., elementwise averaging).
plot(project(f,FunctionSpace(mesh,"DG",0)))

# Option2: Lumped-mass projection to CG1, which remains monotone.
def lumpedProject(f):
    v = TestFunction(V)
    lhs = assemble(inner(Constant(1.0),v)*dx)
    rhs = assemble(inner(f,v)*dx)
    u = Function(V)
    as_backend_type(u.vector())\
        .vec().pointwiseDivide(as_backend_type(rhs).vec(),
                               as_backend_type(lhs).vec())
    return u
plot(lumpedProject(f))

from matplotlib import pyplot as plt
plt.autoscale()
plt.show()