"""
FEniCS tutorial demo program: Linear elastic problem.

  -div(sigma(u)) = f

The model is used to simulate an elastic beam clamped at
its left end and deformed under a boundary load equal to
its own weight.

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
F = -rho*L*W*W*g   # Total force, [N]
traction = F/(W*W) # Traction, [N/m^2]

# PVC
E = 3e9            # Youngs modulus, [Pa]
nu = 0.38          # Poissons ration, [1]
mu = E/(2*(1+nu))  # Shear modulus, [Pa]
lambda_ = E*nu/((1+nu)*(1-2*nu)) # lambda is a reserved word in python

I = W**4/12        # Moment of inertia
print('Euler beam max deflection: ',abs(F*L**3/3/E/I))

# Create mesh and define function space
mesh = BoxMesh.create([Point(0, 0, 0), Point(L, W, W)], [50,10,10], CellType.Type.hexahedron)
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

# Define strain and stress
def epsilon(u):
    return 0.5*(grad(u)+grad(u).T)

def sigma(u):
    eps=epsilon(u)
    return lambda_*tr(eps)*Identity(d)+2*mu*eps

# Define variational problem
u = TrialFunction(V)
d = u.geometric_dimension()  # space dimension
v = TestFunction(V)
f = Constant((0, 0, 0))      # No body force
T = Constant((0, 0, traction))
a = inner(sigma(u), epsilon(v))*dx
L = dot(f, v)*dx + dot(T, v)*ds(1) # Traction only on boundary 1

# Print number of unknowns in equation system
print('DOFs: %i'%V.dim())

# Compute solution
u = Function(V,name='Displacement') # 'name' shown in paraview
solve(a == L, u, bc)

# Calculate stress
s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d)  # deviatoric stress
von_Mises = sqrt(3./2*inner(s, s))
V = FunctionSpace(mesh, 'P', 1)
von_Mises = project(von_Mises, V)
# The first string is the name (shown in paraview),
# the second is a description.
von_Mises.rename('von Mises','von Mises stress')

# Calculate magnitude of displacement
u_magnitude = sqrt(dot(u, u))
u_magnitude = project(u_magnitude, V)
u_magnitude.rename('magnitude','magnitude')
print('min/max u:',
      u_magnitude.vector().min(),
      u_magnitude.vector().max())

# Save solution to file in VTK format
File('elasticity_traction/displacement.pvd') << u
File('elasticity_traction/von_mises.pvd') << von_Mises
File('elasticity_traction/magnitude.pvd') << u_magnitude

