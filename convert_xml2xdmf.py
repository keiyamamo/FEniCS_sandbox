from fenics import *

mesh = Mesh("eccentric_stenosis.xml.gz")

File("mesh.pvd") << mesh

