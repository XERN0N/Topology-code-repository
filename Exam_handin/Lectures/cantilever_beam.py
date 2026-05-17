from fenics import *
from fenics_adjoint import *

class CantileverBeam:
    def __init__(self,L,Wy,Wz,E,nu,F,density,mesh_size,cell_type=CellType.Type.quadrilateral):
        # Use cell_type=CellType.Type.hexahedron for 3D
        # Inputs:
        #   L: x Length, [m]
        #   Wy: y width, [m]
        #   Wz: z width, [m]
        #   E: Youngs modulus, [Pa]
        #   nu: Poissons ration, [1]
        #   F: Force, [N]
        
        self.traction = F/(Wy*Wz)   # Traction, [N/m^2]
        g=9.81                      # Gravitational acceleration, [m/s^2]

        self.E = E
        self.nu = nu
        self.mask=1                 # Can be used to mask out part of the geometry in the body force
        self.mesh_size=mesh_size
        
        if cell_type==CellType.Type.quadrilateral:
            self.dim=2
        else:
            self.dim=3
            
        # Create mesh and define function space
        if self.dim==2:
            self.mesh = RectangleMesh.create([Point(0, 0), Point(L, Wy)], [int(L/mesh_size), int(Wy/mesh_size)], cell_type)
            self.f = Constant((0, -density*g))         # Body force
            self.T = Constant((0, self.traction))
        else:
            self.mesh = BoxMesh.create([Point(0, 0, 0), Point(L, Wy, Wz)], [int(L/mesh_size), int(Wy/mesh_size), int(Wz/mesh_size)], cell_type)
            self.f = Constant((0, -density*g, 0))      # Body force
            self.T = Constant((0, self.traction,0))
        self.V = VectorFunctionSpace(self.mesh, 'P', 1)

        # Define boundary condition
        tol = 1E-10

        class clamped_boundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] < tol

        class load_boundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and abs(x[0]-L)<tol

        if self.dim==2:
            self.boundary_parts = MeshFunction("size_t", self.mesh, 1)
        else:
            self.boundary_parts = MeshFunction("size_t", self.mesh, 2)
        self.boundary_parts.set_all(0)
        load_boundary().mark(self.boundary_parts,1)    # Load on boundary 1
        clamped_boundary().mark(self.boundary_parts,2) # Fixed on boundary 2

        # Define new surface measure with marked surfaces
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundary_parts)
        if self.dim==2:
            self.bc = DirichletBC(self.V, Constant((0, 0)), self.boundary_parts, 2)
        else:
            self.bc = DirichletBC(self.V, Constant((0, 0, 0)), self.boundary_parts, 2)

        # Solver
        self.useIterativeSolver=False
        self.KrylovSolver="gmres"
        self.KrylovPrecon="hypre_amg"

    # Define strain and stress
    def epsilon(self,u):
        return 0.5*(grad(u)+grad(u).T)

    def sigma(self,u,eps=None,E=None):
        if eps is None:
            eps=self.epsilon(u)
        if E is None:
            E=self.E
        mu = E/(2*(1+self.nu))
        lambda_ = E*self.nu/((1+self.nu)*(1-2*self.nu))
        return lambda_*tr(eps)*Identity(self.dim)+2*mu*eps

    def solve(self):
        # Define variational problem
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        a = inner(self.sigma(u), self.epsilon(v))*dx
        L = self.mask*dot(self.f, v)*dx + dot(self.T, v)*self.ds(1) # body force with mask, traction only on boundary 1

        # Compute solution
        u = Function(self.V,name='Displacement') # 'name' shown in paraview
        if self.useIterativeSolver:
            solve(a == L, u, self.bc, solver_parameters={"linear_solver": self.KrylovSolver, "preconditioner": self.KrylovPrecon})
        else:
            solve(a == L, u, self.bc, solver_parameters={'linear_solver': 'mumps'})
        return u

    def vonMises(self,u):
        stress=self.sigma(u)
        s = stress - (1./3)*tr(stress)*Identity(self.dim)  # deviatoric stress
        von_Mises = sqrt(3./2*inner(s, s))
        Vc = FunctionSpace(self.mesh, 'P', 1)
        von_Mises = project(von_Mises, Vc)
        von_Mises.rename('von Mises','von Mises stress')
        return von_Mises

    def gradient(self,u,rho,drho):
        eps=self.epsilon(u)

        dE=derivative(self.E,rho,drho)
        mu = dE/(2*(1+self.nu))
        lambda_ = dE*self.nu/((1+self.nu)*(1-2*self.nu))
        dSigma=lambda_*tr(eps)*Identity(self.dim)+2*mu*eps
        
        g=assemble(-1/2*inner(dSigma,eps)*dx)
        return g

if __name__ == "__main__":
    set_log_level(30)
    rank=MPI.rank(MPI.comm_world)
    
    # Main function if class is not included
    density = 1380     # Density, [kg/m^3] 
    L = 1              # Length, [m]
    W = 0.1            # Width, [m]
    g = 9.81
    F = -density*L*W*W*g   # Total force, [N]
    E = 3e9            # Youngs modulus, [Pa]
    nu = 0.38          # Poissons ration, [1]
    mesh_size=W/9      # Mesh size
    beam=CantileverBeam(L,W,W,E,nu,F,density,mesh_size)
    u=beam.solve()

    # Calculate stress
    von_Mises=beam.vonMises(u)

    # Calculate magnitude of displacement
    V = FunctionSpace(beam.mesh, 'P', 1)
    u_magnitude = sqrt(dot(u, u))
    u_magnitude = project(u_magnitude, V)
    u_magnitude.rename('magnitude','magnitude')
    umin=u_magnitude.vector().min()
    umax=u_magnitude.vector().max()
    if rank==0:
        print('min/max u:', umin, umax)

    # Save solution to file in VTK format
    File('cantilever_beam/displacement.pvd') << u
    File('cantilever_beam/von_mises.pvd') << von_Mises
    File('cantilever_beam/magnitude.pvd') << u_magnitude


    # Test adjoint gradient against fenics_adjoint
    
    parameters['krylov_solver']['absolute_tolerance'] =1e-14
    parameters['krylov_solver']['relative_tolerance'] =1e-14

    from filter_and_project import FilterAndProject
    beta=4
    eta=0.4
    FP=FilterAndProject(mesh_size,beam.mesh)
    
    rho=interpolate( Expression('abs(0.5*sin(x[0]+x[1])+0.35)', degree=1) ,FP.Vd)
    #rho=interpolate(Constant(0.5),FP.Vd)
    
    from pyadjoint.tape import annotate_tape
    while not annotate_tape():
        continue_annotation() # Run continue until we actually start annotating...
    tape=get_working_tape()
    tape.clear_tape()

    rho_tilde=FP.filter(rho)
    rho_bar,rho_tilde_u=FP.Heavi_project(beta,eta,rho_tilde)

    E_min=E/1e6
    p=3
    beam.E=E_min+(E-E_min)*rho_bar**p
    #beam.E=E_min+(E-E_min)*rho**p
    u=beam.solve()
    epsilon=beam.epsilon(u)
    sigma=beam.sigma(u,epsilon)
    
    m=Control(rho)
    J=0.5*inner(sigma,epsilon)*dx
    Volfrac=0.4
    VolMax=W*W*L
    V=(rho_bar-Volfrac)/VolMax*dx
    J=assemble(J)
    V=assemble(V)
    Jhat = ReducedFunctional(J,m)
    Vhat = ReducedFunctional(V,m)
    while annotate_tape():
        pause_annotation() # Run pause until we actually pause annotating...

    Jv=Jhat(rho)
    Vv=Vhat(rho)
    dJ=Jhat.derivative().vector().get_local() # fenics_adjoint gradient of J
    dV=Vhat.derivative().vector().get_local() # fenics_adjoint gradient of V

    epsilon=beam.epsilon(u)
    sigma=beam.sigma(u,epsilon,E=1)
    dcost=FP.project_Vd( p*(E-E_min)*rho_bar**(p-1)*(-1/2*inner(sigma,epsilon)) )
    dbar=FP.dHeavi(beta,eta,rho_tilde_u)*dcost
    dJ_adjoint=FP.filter_gradient(dbar,rho) # adjoint gradient of J
    
    dvol=FP.dHeavi(beta,eta,rho_tilde_u)/VolMax
    dV_adjoint=FP.filter_gradient(dvol,rho) # adjoint gradient of V

    if rank==0:
        delta=1e-4
        l=rho.vector().get_local()
        l[0]+=delta
        rho.vector().set_local( l )
    rho.vector().apply('')
    Jv1=Jhat(rho)
    if rank==0:
        print('COST FUNCTION GRADIENT COMPONENT rho[0] (always on CPU 0) CALCULATED USING FINITE DIFFERENCE:', (Jv1-Jv)/delta)
    
    import numpy as np
    from petsc4py import PETSc
    PETSc.Sys.syncFlush()
    
    for r in range(0,MPI.size(MPI.comm_world)):
        if rank==r:
            # PETSc.Sys.syncPrint is useful for synchronizing the printing on multiple CPUs
            PETSc.Sys.syncPrint('COST FUNCTION on CPU',rank)
            PETSc.Sys.syncPrint('   Maximum of absolute value of fenics_adjoint gradient:', np.max( np.abs(dJ) ))
            PETSc.Sys.syncPrint('   Maximum of absolute value of adjoint gradient       :', np.max( np.abs(dJ_adjoint.get_local()) ))
            PETSc.Sys.syncPrint('   Minimum of absolute value of fenics_adjoint gradient:', np.min( np.abs(dJ) ))
            PETSc.Sys.syncPrint('   Minimum of absolute value of adjoint gradient       :', np.min( np.abs(dJ_adjoint.get_local()) ))

            PETSc.Sys.syncPrint('   First 5 components of fenics_adjoint gradient:', dJ[0:5] )
            PETSc.Sys.syncPrint('   First 5 components of adjoint gradient       :', dJ_adjoint.get_local()[0:5] )

            PETSc.Sys.syncPrint('   Maximum absolute difference between the two gradients:',np.max( np.abs(dJ-dJ_adjoint.get_local()) ))

            PETSc.Sys.syncPrint('VOLUME FUNCTION on CPU',rank)
            PETSc.Sys.syncPrint('   Maximum of absolute value of fenics_adjoint gradient:', np.max( np.abs(dV) ))
            PETSc.Sys.syncPrint('   Maximum of absolute value of adjoint gradient       :', np.max( np.abs(dV_adjoint.get_local()) ))
            PETSc.Sys.syncPrint('   Minimum of absolute value of fenics_adjoint gradient:', np.min( np.abs(dV) ))
            PETSc.Sys.syncPrint('   Minimum of absolute value of adjoint gradient       :', np.min( np.abs(dV_adjoint.get_local()) ))

            PETSc.Sys.syncPrint('   First 5 components of fenics_adjoint gradient:', dV[0:5] )
            PETSc.Sys.syncPrint('   First 5 components of adjoint gradient       :', dV_adjoint.get_local()[0:5] )

            PETSc.Sys.syncPrint('   Maximum absolute difference between the two gradients:', np.max( np.abs(dV-dV_adjoint.get_local()) ))
        PETSc.Sys.syncFlush() # Flushes the print buffer
