from fenics import *

class CantileverLinearBeam:
    def __init__(self,mesh_size,F):
        # F is total force [N]
        # Variables
        L = 1              # Length, [m]
        self.W = 0.1       # Width, [m]
        self.traction = F/self.W**2 # Traction, [N/m^2]

        # PVC
        E = 3e9            # Youngs modulus, [Pa]
        nu = 0.38          # Poissons ration, [1]
        self.mu = E/(2*(1+nu))  # Shear modulus, [Pa]
        self.lambda_ = E*nu/((1+nu)*(1-2*nu)) # lambda is a reserved word in python

        (Nx,Ny,Nz)=( int(L/mesh_size), int(self.W/mesh_size), int(self.W/mesh_size) )
        
        # Create mesh and define function space
        #self.mesh = BoxMesh(Point(0, 0, 0), Point(L, W, W), Nx, Ny, Nz)
        self.mesh = BoxMesh.create([Point(0, 0, 0), Point(L, self.W, self.W)], [Nx,Ny,Nz], CellType.Type.hexahedron)
        self.V = VectorFunctionSpace(self.mesh, 'P', 1)

        # Define boundary condition
        tol = 1E-10

        class clamped_boundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] < tol

        class load_boundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and abs(x[0]-L)<tol

        boundary_parts = MeshFunction("size_t", self.mesh, 2)
        boundary_parts.set_all(0)
        load_boundary().mark(boundary_parts,1)    # Load on boundary 1
        clamped_boundary().mark(boundary_parts,2) # Fixed on boundary 2

        # Define new surface measure with marked surfaces
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=boundary_parts)
        self.bc = DirichletBC(self.V, Constant((0, 0, 0)), boundary_parts, 2)

    def setForce(self,F):
        # Set the surface traction to force F in Newton
        self.traction = F/self.W**2

    # Define strain and stress
    def epsilon(self,u):
        return 0.5*(grad(u)+grad(u).T)

    def sigma(self,u):
        eps=self.epsilon(u)
        d = u.geometric_dimension()  # space dimension
        return self.lambda_*tr(eps)*Identity(d)+2*self.mu*eps

    def solve(self):
        # Define variational problem
        u = TrialFunction(self.V)
        d = u.geometric_dimension()  # space dimension
        v = TestFunction(self.V)
        f = Constant((0, 0, 0))      # No body force
        T = Constant((0, 0, self.traction))
        a = inner(self.sigma(u), self.epsilon(v))*dx
        L = dot(f, v)*dx + dot(T, v)*self.ds(1) # Traction only on boundary 1

        # Print number of unknowns in equation system
        print('DOFs: %i'%self.V.dim())

        # Compute solution
        u = Function(self.V,name='Displacement') # 'name' shown in paraview
        solve(a == L, u, self.bc, solver_parameters={"linear_solver": "cg", "preconditioner": "hypre_amg"})
        return u

    def vonMises(self,u):
        # Calculate stress
        V = FunctionSpace(self.mesh, 'P', 1)
        d = u.geometric_dimension()  # space dimension
        s = self.sigma(u) - (1./3)*tr(self.sigma(u))*Identity(d)  # deviatoric stress
        von_Mises = sqrt(3./2*inner(s, s))
        von_Mises = project(von_Mises, V, solver_type="cg",preconditioner_type="hypre_amg")
        # The first string is the name (shown in paraview),
        # the second is a description.
        von_Mises.rename('von Mises','von Mises stress')
        return von_Mises

    def norm_u(self,u):
        V = FunctionSpace(self.mesh, 'P', 1)
        u_magnitude = sqrt(dot(u, u))
        u_magnitude = project(u_magnitude, V, solver_type="cg",preconditioner_type="hypre_amg")
        u_magnitude.rename('magnitude','magnitude')
        return u_magnitude
        

if __name__ == "__main__":
    # This part will be executed if the file is run as
    # python3 cantileverdLinearBeam.py
    #
    beam=CantileverLinearBeam(0.025,3e4)
    u=beam.solve()
    von_Mises=beam.vonMises(u)
    u_magnitude=beam.norm_u(u)
    print('min/max u:',
          u_magnitude.vector().min(),
          u_magnitude.vector().max())
    # Save solution to file in VTK format
    File('cantileverLinearBeam/displacement.pvd') << u
    File('cantileverLinearBeam/von_mises.pvd') << von_Mises
    File('cantileverLinearBeam/magnitude.pvd') << u_magnitude

