import gmsh
import numpy as np
import sys

gmsh.initialize()
gmsh.model.add("rectangle_circle")

gmsh.model.occ.addDisk(0.2, 1.3, 0, 0.05, 0.05, 2)
gmsh.model.occ.synchronize()



# bottom = gmsh.model.addPhysicalGroup(1, [1], 1)
# gmsh.model.setPhysicalName(1, bottom, "bottom")
# top = gmsh.model.addPhysicalGroup(1, [4], 2)
# gmsh.model.setPhysicalName(1, top, "top")
# side_wall = gmsh.model.addPhysicalGroup(1, [2,3], 3)
# gmsh.model.setPhysicalName(1, side_wall, "side_wall")
solid_boundary = gmsh.model.addPhysicalGroup(1, [1], 1)
gmsh.model.setPhysicalName(1, solid_boundary, "solid_boundary")
# fluid = gmsh.model.addPhysicalGroup(2, [3], 1)
# gmsh.model.setPhysicalName(2, fluid, "fluid")
# solid = gmsh.model.addPhysicalGroup(2, [2], 1)
# gmsh.model.setPhysicalName(2, solid, "solid")
domain = gmsh.model.addPhysicalGroup(2, [2], 1)
gmsh.model.setPhysicalName(2, domain, "domain")

gmsh.option.setNumber("Mesh.MeshSizeMin", 0.002)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.02)
gmsh.model.occ.synchronize()

gmsh.model.mesh.generate(2)
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.write("circle.msh")
gmsh.finalize() 

    