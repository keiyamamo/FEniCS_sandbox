from dolfin import *
import sys




def meshsize(meshname):
    print( meshname)
    mesh = Mesh(meshname[0])
    V = FunctionSpace(mesh, "DG", 0)
    u = TestFunction(V)
    v = TrialFunction(V)
    cell = Cell(mesh, 0)

    cvol = CellVolume(mesh)
    #print(type(cvol))
    #vol = project(cvol, V, solver_type='cg')
    #v = vol.vector().get_local()

    h = ((12/sqrt(2))*(cvol))**.33333333

    A = assemble(inner(u,v)*dx)
    b = assemble(h*u*dx)
    
    h = Function(V)

    print( type(A), type(b), type(h))

    solve(A, h.vector(), b, "gmres")
#    plot(h)
#    interactive()
    meshfilename = File(meshname[0]+"_nodespacing.pvd")
    meshfilename << h
if __name__ == '__main__':
    args = sys.argv[1:]
    meshsize(args)