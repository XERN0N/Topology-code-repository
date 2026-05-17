from fenics import *
from fenics_adjoint import *
from cantilever_beam import CantileverBeam

class MBB(CantileverBeam):
    def __init__(self,L,Wy,Wz,E,nu,F,density,mesh_size,cell_type=CellType.Type.quadrilateral):
        # Use cell_type=CellType.Type.hexahedron for 3D
        # Inputs:
        #   L: x Length, [m]
        #   Wy: y width, [m]
        #   Wz: z width, [m]
        #   E: Youngs modulus, [Pa]
        #   nu: Poissons ration, [1]
        #   F: Force, [N]
        CantileverBeam.__init__(self,L,Wy,Wz,E,nu,F,density,mesh_size,cell_type)
        
        # Define new boundary condition
        self.boundary_parts.set_all(0)

        class XSymmetry_boundary(SubDomain):
            def inside(self,x,on_boundary):
                return on_boundary and near(x[0],0)
        class Roll_boundary(SubDomain):
            def __init__(self,Lx,Ly,rollersize):
                self.Lx=Lx
                self.Ly=Ly
                self.RollerSize=rollersize
                SubDomain.__init__(self)
            def inside(self,x,on_boundary):
                tol = 1E-10
                return on_boundary and near(x[1],0) and (x[0]>=(self.Lx-self.RollerSize-tol) and x[0]<=(self.Lx+tol))
        class Load_boundary(SubDomain):
            def __init__(self,Lx,Ly,loadsize):
                self.Lx=Lx
                self.Ly=Ly
                self.LoadSize=loadsize
                SubDomain.__init__(self)
            def inside(self,x,on_boundary):
                tol = 1E-10
                return on_boundary and near(x[1],self.Ly) and (x[0]>=(0.0-tol) and x[0]<=(self.LoadSize+tol))

        loadsize=self.mesh_size*2
        load_boundary=Load_boundary(L,Wy,loadsize)
        load_boundary.mark(self.boundary_parts,1)
        xsym_boundary=XSymmetry_boundary()
        xsym_boundary.mark(self.boundary_parts,2)
        roll_boundary=Roll_boundary(L,Wy,self.mesh_size)
        roll_boundary.mark(self.boundary_parts,3)

        # Redefine traction size
        self.traction = self.traction*(Wy*Wz)/loadsize

        self.ds=Measure('ds',domain=self.mesh,subdomain_data=self.boundary_parts)
        if self.dim==2:
            self.bc=[DirichletBC(self.V.sub(0), 0, self.boundary_parts,2),DirichletBC(self.V.sub(1), 0, self.boundary_parts,3)]
        else:
            self.bc=[DirichletBC(self.V.sub(0), 0, self.boundary_parts,2),DirichletBC(self.V.sub(1), 0, self.boundary_parts,3),
                     DirichletBC(self.V.sub(2), 0, self.boundary_parts,3)]

if __name__ == "__main__":
    # Main function, if file is executed directly
    density = 1380     # Density, [kg/m^3] 
    L = 1              # Length, [m]
    W = 0.1            # Width, [m]
    g = 9.81
    F = -density*L*W*W*g   # Total force, [N]
    E = 3e9            # Youngs modulus, [Pa]
    nu = 0.38          # Poissons ration, [1]
    mesh_size=W/10     # Mesh size
    beam=MBB(L,W,W,E,nu,F,density,mesh_size,cell_type=CellType.Type.hexahedron)
    File('mbb/boundary_parts.pvd') << beam.boundary_parts
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
    File('mbb/displacement.pvd') << u
    File('mbb/von_mises.pvd') << von_Mises
    File('mbb/magnitude.pvd') << u_magnitude

    # Test adjoint gradient against fenics_adjoint
    
    parameters['krylov_solver']['absolute_tolerance'] =1e-14
    parameters['krylov_solver']['relative_tolerance'] =1e-14
    
    from pyadjoint.tape import annotate_tape
    while not annotate_tape():
        continue_annotation() # Run continue until we actually start annotating...
    tape=get_working_tape()
    tape.clear_tape()
    Vd=FunctionSpace(beam.mesh,'DG',0)
    rho=interpolate(Constant(0.5),Vd)
    beam.E=E*rho
    u=beam.solve()
    eps=beam.epsilon(u)
    sig=beam.sigma(u,eps)
    m=Control(rho)
    J=0.5*inner(sig,eps)*dx
    Volfrac=0.4
    VolMax=W*W*L
    V=(rho-Volfrac)/VolMax*dx
    J=assemble(J)
    V=assemble(V)
    Jhat = ReducedFunctional(J,m)
    Vhat = ReducedFunctional(V,m)
    while annotate_tape():
        pause_annotation() # Run pause until we actually pause annotating...

    Jv=Jhat(rho)
    Vv=Vhat(rho)
    dJ=Jhat.derivative().vector().get_local() # fenics_adjoint gradient
    dV=Vhat.derivative().vector().get_local()

    drho=TestFunction(Vd)
    dJ2=beam.gradient(u,rho,drho) # Ajdoint gradient

    import numpy as np
    print('Maximum of absolute value of fenics_adjoint gradient:',np.max(np.abs( dJ )))
    print('Maximum of absolute value of adjoint gradient       :',np.max(np.abs( dJ2.get_local() )))

    print('Minimum of absolute value of fenics_adjoint gradient:',np.min(np.abs( dJ )))
    print('Minimum of absolute value of adjoint gradient       :',np.min(np.abs( dJ2.get_local() )))
    
    print('First few components of fenics_adjoint gradient:', dJ[0:5] )
    print('First few components of adjoint gradient       :', dJ2.get_local()[0:5] )
    
    print('Maximum absolute difference between the two gradients:',np.max( np.abs(dJ-dJ2.get_local()) ))
