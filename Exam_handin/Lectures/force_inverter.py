from fenics import *
from fenics_adjoint import *

class ForceInverter():
    def __init__(self,L,W,E,nu,F,density,mesh_size,comm=None,meshfile=None):
        # 2D force inverter
        # Inputs:
        #   L: x Length, [m]
        #   W: half y width, [m]
        #   E: Youngs modulus, [Pa]
        #   nu: Poissons ration, [1]
        #   F: Force, [N]
        g=9.81                      # Gravitational acceleration, [m/s^2]

        if comm is None:
            self.comm=MPI.comm_world
        else:
            self.comm=comm

        self.L=L
        self.W=W
        self.E = E
        self.nu = nu
        self.mask=1               # Can be used to mask out part of the geometry in the body force
        self.mesh_size=mesh_size
        self.dim=2
        
        # Create mesh and define function space
        if meshfile is None:
            self.mesh = RectangleMesh.create([Point(0, 0), Point(L, W)], [int(L/mesh_size), int(W/mesh_size)], CellType.Type.quadrilateral)
        else:
            self.mesh = Mesh(self.comm)
            with XDMFFile(self.comm,meshfile) as f:
                f.read(self.mesh)

        self.V = VectorFunctionSpace(self.mesh, 'P', 1)
        
        self.boundary_parts = MeshFunction("size_t", self.mesh, 1)
        # Define new boundary condition
        self.boundary_parts.set_all(0)

        self.tol=1e-8
        self.LoadSize=W/5
        self.ClampSize=W/10
        self.MoveSize=W/5
        
        class YSymmetry_boundary(SubDomain):
            def inside(self,x,on_boundary):
                return on_boundary and near(x[1],0)
        class generic_boundary_marker(SubDomain):
            def __init__(self,f):
                self.f=f
                SubDomain.__init__(self)
            def inside(self,x,on_boundary):
                return on_boundary and self.f(x)
            
        load_boundary=generic_boundary_marker(self.insideLoad)
        load_boundary.mark(self.boundary_parts,1)
        clamped_boundary=generic_boundary_marker(self.insideClamped)
        clamped_boundary.mark(self.boundary_parts,3)
        move_boundary=generic_boundary_marker(self.insideMove)
        move_boundary.mark(self.boundary_parts,4)
        ysym_boundary=YSymmetry_boundary()
        ysym_boundary.mark(self.boundary_parts,2) # Must be marked last to overwrite other markings...

        self.traction = F/(W*W)   # Traction, [N/m^2]
        self.f = Constant((0, -density*g))         # Body force
        self.T = Constant((self.traction,0))
        self.spring_kin=self.traction/self.LoadSize  # Spring on input  (load boundary)
        self.spring_kout=0.25*self.spring_kin        # Spring on output (move boundary)
        
        self.ds=Measure('ds',domain=self.mesh,subdomain_data=self.boundary_parts)
        self.bc=[DirichletBC(self.V.sub(1), 0, self.boundary_parts,2),DirichletBC(self.V, (0,0), self.boundary_parts,3)]
        
        # Solver
        self.useIterativeSolver=True # Let's try an iterative solver...
        self.KrylovSolver="gmres"
        self.KrylovPrecon="hypre_amg"

    def insideClamped(self,x):
        return x[0]<self.ClampSize/2 and x[1]>(self.W-self.ClampSize-self.tol)
    def insideLoad(self,x):
        return x[0]<self.LoadSize/2 and x[1]<(self.LoadSize/2+self.tol)
    def insideMove(self,x):
        return x[0]>(self.L-self.MoveSize/2-self.tol) and x[1]<(self.MoveSize/2+self.tol) and x[1]<(self.MoveSize/2+self.tol)
        
    # Define strain and stress
    def epsilon(self,u):
        return 0.5*(grad(u)+grad(u).T)

    def sigma(self,u,eps=None,E=None):
        if eps is None:
            eps=self.epsilon(u)
        if E is None:
            E=self.E
        d = u.geometric_dimension()
        mu = E/(2*(1+self.nu))
        lambda_ = E*self.nu/((1+self.nu)*(1-2*self.nu))
        return lambda_*tr(eps)*Identity(d)+2*mu*eps

    def solve(self):
        # Define variational problem
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        a = inner(self.sigma(u), self.epsilon(v))*dx \
            + self.spring_kin*u[0]*v[0]*self.ds(1) \
            + self.spring_kout*u[0]*v[0]*self.ds(4)
        L = self.mask*dot(self.f, v)*dx + dot(self.T, v)*self.ds(1) # body force with mask, traction only on boundary 1

        # Compute solution
        u = Function(self.V,name='Displacement') # 'name' shown in paraview
        if self.useIterativeSolver:
            solve(a == L, u, self.bc, solver_parameters={"linear_solver": self.KrylovSolver, "preconditioner": self.KrylovPrecon})
        else:
            solve(a == L, u, self.bc, solver_parameters={'linear_solver': 'mumps'})
        return u

    def vonMises(self,u):
        d = u.geometric_dimension()  # space dimension
        stress=self.sigma(u)
        s = stress - (1./3)*tr(stress)*Identity(d)  # deviatoric stress
        von_Mises = sqrt(3./2*inner(s, s))
        Vc = FunctionSpace(self.mesh, 'P', 1)
        von_Mises = project(von_Mises, Vc)
        von_Mises.rename('von Mises','von Mises stress')
        return von_Mises

if __name__ == "__main__":
    # Main function if class is not included
    L = 1              # Length, [m]
    W = 1              # Width, [m]
    g = 9.81
    F = 1e6            # Total force, [N]
    E = 3e9            # Youngs modulus, [Pa]
    nu = 0.38          # Poissons ration, [1]
    mesh_size=W/20     # Mesh size
    beam=ForceInverter(L,W,E,nu,F,0,mesh_size)
    directory='force_inverter/'
    File(directory+'boundary_parts.pvd') << beam.boundary_parts
    u=beam.solve()

    # Calculate stress
    von_Mises=beam.vonMises(u)

    # Calculate magnitude of displacement
    V = FunctionSpace(beam.mesh, 'P', 1)
    u_magnitude = sqrt(dot(u, u))
    u_magnitude = project(u_magnitude, V)
    u_magnitude.rename('magnitude','magnitude')
    print('min/max u:',
          u_magnitude.vector().min(),
          u_magnitude.vector().max())

    # Save solution to file in VTK format
    File(directory+'displacement.pvd') << u
    File(directory+'von_mises.pvd') << von_Mises
    File(directory+'magnitude.pvd') << u_magnitude

