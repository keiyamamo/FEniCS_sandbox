#!/usr/bin/env python
# coding: utf-8

# # Modal analysis of an elastic structure
# 
# This program performs a dynamic modal analysis of an elastic cantilever beam
# represented by a 3D solid continuum. The eigenmodes are computed using the
# **SLEPcEigensolver** and compared against an analytical solution of beam theory. We also discuss the computation of modal participation factors.
# 
# 
# The first four eigenmodes of this demo will look as follows:

# The first two fundamental modes are on top with bending along the weak axis (left) and along
# the strong axis (right), the next two modes are at the bottom.
# 
# ## Implementation
# 
# After importing the relevant modules, the geometry of a beam of length $L=20$
# and rectangular section of size $B\times H$ with $B=0.5, H=1$ is first defined:

# In[1]:


from dolfin import *
import numpy as np

L, B, H = 20., 0.5, 1.

Nx = 200
Ny = int(B/L*Nx)+1
Nz = int(H/L*Nx)+1

mesh = BoxMesh(Point(0.,0.,0.),Point(L,B,H), Nx, Ny, Nz)


# Material parameters and elastic constitutive relations are classical (here we take $\nu=0$) and we also introduce the material density $\rho$ for later definition of the mass matrix:

# In[2]:


E, nu = Constant(1e5), Constant(0.)
rho = Constant(1e-3)

# Lame coefficient for constitutive relation
mu = E/2./(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)

def eps(v):
    return sym(grad(v))
def sigma(v):
    dim = v.geometric_dimension()
    return 2.0*mu*eps(v) + lmbda*tr(eps(v))*Identity(dim)


# Standard `FunctionSpace` is defined and boundary conditions correspond to a fully clamped support at $x=0$.

# In[3]:


V = VectorFunctionSpace(mesh, 'Lagrange', degree=1)
u_ = TrialFunction(V)
du = TestFunction(V)


def left(x, on_boundary):
    return near(x[0],0.)

bc = DirichletBC(V, Constant((0.,0.,0.)), left)


# The system stiffness matrix $[K]$ and mass matrix $[M]$ are respectively obtained from assembling the corresponding variational forms

# In[4]:


k_form = inner(sigma(du),eps(u_))*dx
l_form = Constant(1.)*u_[0]*dx
K = PETScMatrix()
b = PETScVector()
assemble_system(k_form, l_form, bc, A_tensor=K, b_tensor=b)

m_form = rho*dot(du,u_)*dx
M = PETScMatrix()
assemble(m_form, tensor=M)


# Matrices $[K]$ and $[M]$ are first defined as `PETScMatrix` and forms are assembled into it to ensure that they have the right type.
# Note that boundary conditions have been applied to the stiffness matrix using `assemble_system` so as to preserve symmetry (a dummy `l_form` and right-hand side vector have been introduced to call this function).
# 
# Modal dynamic analysis consists in solving the following generalized eigenvalue problem $[K]\{U\}=\lambda[M]\{U\}$ where the eigenvalue is related to the eigenfrequency $\lambda=\omega^2$. This problem can be solved using the `SLEPcEigenSolver`.

# In[5]:


eigensolver = SLEPcEigenSolver(K, M)
eigensolver.parameters['problem_type'] = 'gen_hermitian'
eigensolver.parameters['spectral_transform'] = 'shift-and-invert'
eigensolver.parameters['spectral_shift'] = 0.


# The problem type is specified to be a generalized eigenvalue problem with Hermitian matrices. By default, SLEPc computes the largest eigenvalues. Here we instead look for the smallest eigenvalues (they should all be real). A  spectral transform is therefore performed using the keyword `shift-invert` i.e. the original problem is transformed into an equivalent problem with eigenvalues given by $\dfrac{1}{\lambda - \sigma}$ instead of $\lambda$ where $\sigma$ is the value of the spectral shift. It is therefore much easier to compute eigenvalues close to $\sigma$ i.e. close to $\sigma = 0$ in the present case. Eigenvalues are then transformed back by SLEPc to their original value $\lambda$.
# 
# 
# We now ask SLEPc to extract the first 6 eigenvalues by calling its solve function and extract the corresponding eigenpair (first two arguments of `get_eigenpair` correspond to the real and complex part of the eigenvalue, the last two to the real and complex part of the eigenvector).

# In[6]:


N_eig = 6   # number of eigenvalues
print("Computing {} first eigenvalues...".format(N_eig))
eigensolver.solve(N_eig)

# Exact solution computation
from scipy.optimize import root
from math import cos, cosh
falpha = lambda x: cos(x)*cosh(x)+1
alpha = lambda n: root(falpha, (2*n+1)*pi/2.)['x'][0]

# Set up file for exporting results
file_results = XDMFFile("modal_analysis.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

eigenmodes = []
# Extraction
for i in range(N_eig):
    # Extract eigenpair
    r, c, rx, cx = eigensolver.get_eigenpair(i)

    # 3D eigenfrequency
    freq_3D = sqrt(r)/2/pi

    # Beam eigenfrequency
    if i % 2 == 0: # exact solution should correspond to weak axis bending
        I_bend = H*B**3/12.
    else:          #exact solution should correspond to strong axis bending
        I_bend = B*H**3/12.
    freq_beam = alpha(i/2)**2*sqrt(float(E)*I_bend/(float(rho)*B*H*L**4))/2/pi

    print("Solid FE: {:8.5f} [Hz]   Beam theory: {:8.5f} [Hz]".format(freq_3D, freq_beam))

    # Initialize function and assign eigenvector
    eigenmode = Function(V,name="Eigenvector "+str(i))
    eigenmode.vector()[:] = rx
    
    eigenmodes.append(eigenmode)


# The beam analytical solution is obtained using the eigenfrequencies of a clamped beam in bending given by $\omega_n = \alpha_n^2\sqrt{\dfrac{EI}{\rho S L^4}}$ where :math:`S=BH` is the beam section, :math:`I` the bending inertia and $\alpha_n$ is the solution of the following nonlinear equation:
# 
# \begin{equation}
# \cos(\alpha)\cosh(\alpha)+1 = 0
# \end{equation}
# 
# the solution of which can be well approximated by $(2n+1)\pi/2$ for $n\geq 3$.
# 
# Since the beam possesses two bending axis, each solution to the previous equation is
# associated with two frequencies, one with bending along the weak axis ($I=I_{\text{weak}} = HB^3/12$)
# and the other along the strong axis ($I=I_{\text{strong}} = BH^3/12$). Since $I_{\text{strong}} = 4I_{\text{weak}}$ for the considered numerical values, the strong axis bending frequency will be twice that corresponding to bending along the weak axis. The solution $\alpha_n$ are computed using the
# `scipy.optimize.root` function with initial guess given by $(2n+1)\pi/2$.
# 
# With `Nx=400`, we obtain the following comparison between the FE eigenfrequencies and the beam theory eigenfrequencies :
# 
# 
# | Mode | Solid FE [Hz] | Beam theory [Hz] |
# | --- | ------ | ------- |
# | 1 |  2.04991 |  2.01925|
# | 2 |  4.04854 |  4.03850|
# | 3 | 12.81504 | 12.65443|
# | 4 | 25.12717 | 25.30886|
# | 5 | 35.74168 | 35.43277|
# | 6 | 66.94816 | 70.86554|
# 

# ## Modal participation factors
# 
# In this section we show how to compute modal participation factors for a lateral displacement in the $Y$ direction. Modal participation factors are defined as:
# 
# \begin{equation}
# q_i = \{\xi_i\}[M]\{U\}
# \end{equation}
# 
# where $\{\xi_i\}$ is the i-th eigenmode and $\{U\}$ is a vector of unit displacement in the considered direction. The corresponding effective mass is given by:
# 
# \begin{equation}
# m_{\text{eff},i} = \left(\dfrac{\{\xi_i\}[M]\{U\}}{\{\xi_i\}[M]\{\xi_i\}}\right)^2 = \left(\dfrac{q_i}{m_i}\right)^2
# \end{equation}
# 
# where $m_i$ is the modal mass which is in general equal to 1 for eigensolvers which adopt the mass matrix normalization convention.
# 
# With `FEniCS`, the modal participation factor can be easily computed by taking the `action` of the mass form with both the mode and a unit displacement function. Let us now print the corresponding effective mass of the 6 first modes.

# In[8]:


u = Function(V, name="Unit displacement")
u.interpolate(Constant((0, 1, 0)))
combined_mass = 0
for i, xi in enumerate(eigenmodes):
    qi = assemble(action(action(m_form, u), xi))
    mi = assemble(action(action(m_form, xi), xi))
    meff_i = (qi / mi) ** 2
    total_mass = assemble(rho * dx(domain=mesh))

    print("-" * 50)
    print("Mode {}:".format(i + 1))
    print("  Modal participation factor: {:.2e}".format(qi))
    print("  Modal mass: {:.4f}".format(mi))
    print("  Effective mass: {:.2e}".format(meff_i))
    print("  Relative contribution: {:.2f} %".format(100 * meff_i / total_mass))

    combined_mass += meff_i

print(
    "\nTotal relative mass of the first {} modes: {:.2f} %".format(
        N_eig, 100 * combined_mass / total_mass
    )
)


# We see that the first, third and fifth mode are those having the larger participation which is consistent with the fact that they correspond to horizontal vibrations of the beam.

# In[ ]:




