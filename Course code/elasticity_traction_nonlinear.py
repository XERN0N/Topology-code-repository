"""
FEniCS tutorial demo program: Non-Linear elastic problem.

The model is used to simulate an elastic beam clamped at
its left end and deformed under a boundary load equal to
1000x its own weight. Green-Lagrange strain is used.

To visualize the solution use paraview:
1. Open displacement.pvd and von_mises.pvd
2. Select both in the pipeline browser and apply the Append Attributes filter.
3. Select the bottom AppendAttributes and apply the WarpByVector filter (the beam button).
"""
from fenics import *

# Variables
rho = 1380         # Density, [kg/m^3] 
L = 1              # Length, [m]
W = 0.1            # Width, [m]
g = 9.81           # Gravitational acceleration, [m/s^2]
F = -1000*rho*L*W*W*g   # Total force, [N]
traction = F/(W*W) # Traction, [N/m^2]

# PVC
E = 3e9            # Youngs modulus, [Pa]
nu = 0.38          # Poissons ration, [1]
mu = E/(2*(1+nu))  # Shear modulus, [Pa]
lambda_ = E*nu/((1+nu)*(1-2*nu)) # lambda is a reserved word in python

I = W**4/12        # Moment of inertia
print('Euler beam max deflection: ',abs(F*L**3/3/E/I))

# Create mesh and define function space
mesh = BoxMesh.create([Point(0, 0, 0), Point(L, W, W)],
                      [50,10,10], CellType.Type.hexahedron)
V = VectorFunctionSpace(mesh, 'P', 1)

# Define boundary condition
tol = 1E-10

class clamped_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] < tol

class load_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0]-L)<tol

boundary_parts = MeshFunction("size_t", mesh, 2)
boundary_parts.set_all(0)
load_boundary().mark(boundary_parts,1)    # Load on boundary 1
clamped_boundary().mark(boundary_parts,2) # Fixed on boundary 2

# Save markings for inspection in paraview
File('elasticity_traction/boundary_parts.pvd') << boundary_parts

# Define new surface measure with marked surfaces
ds = Measure('ds', domain=mesh, subdomain_data=boundary_parts)
bc = DirichletBC(V, Constant((0, 0, 0)), boundary_parts, 2)

# Define variational problem
u = Function(V,name='Displacement')
du = TrialFunction(V)
v = TestFunction(V)
f = Constant((0, 0, 0))      # No body force
T = Constant((0, 0, traction))

# Linear strain
#eps = (grad(u)+grad(u).T)/2
# Green-Lagrange strain
eps = (grad(u)+grad(u).T+dot(grad(u),grad(u).T))/2

# Stored strain energy density
Psi = mu*tr(eps*eps)+lambda_/2*tr(eps)**2

# Total potential energy
Pi = Psi*dx - dot(f, u)*dx - dot(T, u)*ds(1)

# Directional derivative about u in the direction of v
# give the nonlinear equations F(u,v)=0
F = derivative(Pi, u, v)

# Compute Jacobian of F for Newton’s method
dF = derivative(F, u, du)

# Create nonlinear variational problem and solve
problem = NonlinearVariationalProblem(F, u, bcs=bc, J=dF)
solver = NonlinearVariationalSolver(problem)
solver.parameters['newton_solver']['absolute_tolerance'] = 1e-6
solver.parameters['newton_solver']['linear_solver'] = 'cg'
solver.parameters['newton_solver']['preconditioner'] = 'sor'
solver.solve()

# Calculate magnitude of displacement
Vmag = FunctionSpace(mesh, 'P', 1)
u_magnitude = project( sqrt(dot(u, u)) , Vmag)
u_magnitude.rename('magnitude','magnitude')
print('min/max u:',
      u_magnitude.vector().min(),
      u_magnitude.vector().max())

# Save solution to file in VTK format
File('elasticity_traction_nonlinear/displacement.pvd') << u
