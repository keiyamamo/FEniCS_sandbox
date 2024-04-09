from fenics import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Geometry and mesh
lpx, lpy = 100, 10
mesh = RectangleMesh(Point(0, 0), Point(lpx, lpy), 10, 10)

V_element = VectorElement("CG", mesh.ufl_cell(), 2)  # Displacement field

# Function space
V = FunctionSpace(mesh, V_element)

# Boundary conditions
def left(x, on_boundary):
    return on_boundary and near(x[0], 0)

def right(x, on_boundary):
    return on_boundary and near(x[0], lpx)

def bottom(x, on_boundary):
    return on_boundary and near(x[1], 0)

# Material parameters for Mooney-Rivlin model
C1, C2,  = 1.113, 0.12 # Example values
mu =10
Kappa = 10*mu

# Test and trial functions
u = Function(V)
du = TrialFunction(V)
v = TestFunction(V)

# Kinematics
d = len(u)
I = Identity(d)
F = I + grad(u)
C = F.T * F
Ic = tr(C)
J = det(F)

# Invariants of the deformation
I1 = tr(C)
I2 = 0.5 * (I1**2 - tr(C * C))

# Mooney-Rivlin energy function
Psi_MR = C1 * (I1 - 3) + C2 * (I2 - 3) - Kappa/2 * (ln(J))**2
Pi = Psi_MR * dx

# Compute first variation of Pi (the derivative of Pi with respect to u in the direction of Test_u)
F_variational = derivative(Pi, u, v)

# Compute Jacobian of F
J_variational = derivative(F_variational, u, du)

# Apply boundary conditions
bc_left = DirichletBC(V.sub(0), Constant(0.0), left)
bc_bottom = DirichletBC(V.sub(1), Constant(0.0), bottom)

E_11_values = []
S_11_values = []
T_11_values = []
stretch_values = []

# Loop over stretch values
for stretch in [1.0 + i * 0.1 for i in range(51)]:
    # Update right boundary condition
    bc_right = DirichletBC(V.sub(0), Constant(stretch * lpx - lpx), right)
    bcs = [bc_left, bc_right, bc_bottom]

    # Solve the problem
    problem = NonlinearVariationalProblem(F_variational, u, bcs, J=J_variational)
    solver = NonlinearVariationalSolver(problem)
    solver.solve()


    # Plot the deformed configuration
    p = plot(u, mode="displacement")
    plt.colorbar(p)
    plt.title(f"Deformed configuration at stretch = {stretch:.2f}")
    plt.savefig(f" MooneyRivlinModel_deformed_configuration_{stretch:.2f}.png")
    plt.close()

    def Cauchy_stress_MR(F, C1, C2):
        I = Identity(2)
        B = F * F.T  
        J = det(F)

        stress_tensor = C1 * I + 2 * C2 * B - (C1 + 2 * C2) * J**(-2/3) * B*B

        return stress_tensor


    def PK1_stress_MR(F, C1, C2):
        I = Identity(2)
        J = det(F)
        B = F * F.T  # Left Cauchy-Green deformation tensor
        FinvT = inv(F).T  # F'nin tersinin transpozesi

        # Mooney-Rivlin modeline göre PK1 gerilme tensörünün hesaplanması
        PK1 = F * (C1 * I + 2 * C2 * B - (C1 + 2 * C2) * J**(-2/3) * B*B)

        return PK1
    
    def PK2_stress(F, P):
        F_inv = inv(F)
        return  F_inv * P
    
    def green_lagrange_strain(F):
        C = F.T * F
        I = Identity(2)
        return 0.5 * (C - I)
    
    F_val = grad(u) + Identity(2)
    P_val = PK1_stress_MR(F_val, C1, C2)

    S_tensor = PK2_stress(F_val, P_val)
    E_tensor = green_lagrange_strain(F_val)
    T_tensor = Cauchy_stress_MR(F_val, C1, C2)

    S_tensor_field = project(S_tensor, TensorFunctionSpace(mesh, "DG", 0))
    E_tensor_field = project(E_tensor, TensorFunctionSpace(mesh, "DG", 0))
    T_tensor_field = project(E_tensor, TensorFunctionSpace(mesh, "DG", 0))

    x_coord = 100  # x coordinate
    y_coord = 5   # y coordinate
    point = Point(x_coord, y_coord)

    S_at_point = S_tensor_field(point)
    E_at_point = E_tensor_field(point)
    T_at_point = T_tensor_field(point)

    print(f"#######################################              STRETCH: {stretch}            ###########################################")
   
    print(f"Cauchy Stress Tensor at ({x_coord}, {y_coord}): {T_at_point}")
    print(f"PK2 Stress Tensor at ({x_coord}, {y_coord}): {S_at_point}")
    print(f"GL Strain Tensor at ({x_coord}, {y_coord}): {E_at_point}")
    print("________________________________________________________________________________________________________________________________")
    print("********************************************************************************************************************************")

    E_11_values.append(E_at_point[0])
    S_11_values.append(S_at_point[0])
    T_11_values.append(T_at_point[0])
    stretch_values.append(stretch)

    plt.figure()
    plt.plot(stretch_values, S_11_values, 'o-')
    plt.xlabel('Stretch (λ)')
    plt.ylabel('S_11')
    plt.title('MooneyRivlinMaterialModel S_11 vs Stretch (λ)')
    plt.grid(True)
    plt.savefig('MooneyRivlinMaterialModel S11 vs Stretch.png')
    plt.show()

    plt.figure()
    plt.plot(stretch_values, T_11_values, 'o-')
    plt.xlabel('stretch')
    plt.ylabel('T_11')
    plt.title('MooneyRivlinMaterialModel T_11 vs Stretch (λ)')
    plt.grid(True)
    plt.savefig('MooneyRivlinMaterialModel T_11 vs Stretch.png')
    plt.show()