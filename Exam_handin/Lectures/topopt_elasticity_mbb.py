from fenics import *
from fenics_adjoint import *
from mbb import MBB
from filter_and_project import FilterAndProject
import numpy as np
from mma import MMA_petsc

class topOpt():
    def __init__(self):
        # MPI rank
        self.rank=MPI.rank(MPI.comm_world)

        # Directory for output
        self.directory='topopt_elasticity_mbb/'

        # Some constants
        density = 1380     # Density, [kg/m^3] 
        L = 1              # Length, [m]
        W = L/3            # Width, [m]
        g = 9.81           # Gravitational acceleration, [m/s^2]
        density = 1380     # Density, [kg/m^3] 
        F = -density*L*W*W*g   # Total force on boundary, [N]
        self.E0 = 3e9      # Youngs modulus, [Pa]
        self.nu = 0.38     # Poissons ration, [1]
        mesh_size=W/50     # Mesh size
        self.Volfrac=0.5
        self.VolMax=L*W
        
        R=W/8/(2*np.sqrt(3))  # Filter Radius ~ W/8
        #R=2*mesh_size/(2*np.sqrt(3))  # Minimum filter Radius
        self.Emin=self.E0/1e6 # Minimum E
        self.pen=3            # SIMP penalization 
        
        self.mech=MBB(L,W,W,self.E0,self.nu,F,0*density,mesh_size)#,cell_type=CellType.Type.hexahedron)
        
        self.FP=FilterAndProject(R,self.mech.mesh)
        self.rho=interpolate(Constant(0.5),self.FP.Vd)
        #self.rho=interpolate( Expression('abs(0.5*sin(x[0]+x[1])+0.35)', degree=1) ,self.FP.Vd) # Enable to try another starting density
        self.rho.rename('rho','rho')
        self.rho_petsc=as_backend_type(self.rho.vector()).vec()  # Pointer to rho's petsc data structure

        # Set up optimizer
        # Number of constraints
        self.ncon=1
        self.setUpOptimizer( self.rho_petsc )
    
    def setUpOptimizer(self,x0):
        """Creates and initializes MMA"""
        self.mma=MMA_petsc(x0,self.ncon,f=self.f,g=self.g,plot_k=self.plot_k)
        self.mma.xmin[:]=0.0
        self.mma.xmax[:]=1.0

        # MMA setup
        self.mma.move=0.5
        self.mma.xtol=0.01
        self.mma.ftol=0.001
        self.mma.lmax=5
        self.mma.kmax=30
        self.mma.kmin=1
        self.mma.it=0
        self.mma.mma_timing=False  

    def Forward(self,beta,eta):
        rho_tilde=self.FP.filter(self.rho)
        rho_bar,_=self.FP.Heavi_project(beta,eta,rho_tilde)
        #rho_tilde=self.rho # Use to "turn off" filtering
        #rho_bar=rho_tilde  # and projection
        E=self.Emin+pow(rho_bar,self.pen)*(self.E0-self.Emin) # SIMP E
        self.mech.E=E
        self.mech.mask=rho_bar
        u=self.mech.solve()
        return (u,rho_bar,rho_tilde)

    def plot_k(self,x,force_plot=False):
        """Used for output during optimization""" 
        if force_plot or self.iter % self.nout == 0:
            if not force_plot:
                self.rho.vector().set_local(x.getArray()[:])
                self.rho.vector().apply('')

            (u,rho_bar,rho_tilde)=self.Forward(self.beta,self.eta)
            
            self.u_fid << u,self.iter
            self.rho_fid << self.rho,self.iter
            self.rho_tilde_fid << rho_tilde,self.iter
            self.rho_bar_fid << rho_bar,self.iter
        self.iter+=1

    def setUpFunctionals(self,beta,eta):
        from pyadjoint.tape import annotate_tape
        while not annotate_tape():
            continue_annotation() # Run continue until we actually start annotating...
        tape=get_working_tape()
        tape.clear_tape()
        # Run forward model to record on tape
        (u,rho_bar,rho_tilde)=self.Forward(beta,eta)
        eps=self.mech.epsilon(u)
        sig=self.mech.sigma(u,eps)
        # Using annotate=False to not record this step, such that J0 does not depend on rho
        if beta==1:
            self.J0=float(assemble(0.5*inner(sig,eps)*dx,annotate=False))
        # Make functionals
        m=Control(self.rho)
        self.J=0.5*inner(sig,eps)/self.J0*dx
        self.V=(rho_bar-self.Volfrac)/self.VolMax*dx
        J=assemble(self.J)
        V=assemble(self.V)
        #self.Jhat = ReducedFunctional(J,m,tape=tape)
        #self.Vhat = ReducedFunctional(V,m,tape=tape)
        # Copy tape, to be able to run optimize such that e.g. mechanics is not solved for when calculating volume.
        tapeJ=tape.copy()
        tapeV=tape.copy()
        self.Jhat = ReducedFunctional(J,m,tape=tapeJ)
        self.Vhat = ReducedFunctional(V,m,tape=tapeV)
        tapeJ.optimize(controls=[m],functionals=[J])
        tapeV.optimize(controls=[m],functionals=[V])
        while annotate_tape():
            pause_annotation() # Run pause until we actually pause annotating...

    def f(self,x):
        """Define function for MMA that calculates the vector function f"""
        self.rho.vector().set_local(x.getArray()[:])
        self.rho.vector().apply('')

        J=self.Jhat(self.rho)
        V=self.Vhat(self.rho)
        return np.array([J,V])

    def g(self,x):
        """Define function for MMA that calculates the gradient (Jacobian) of f"""
        self.rho.vector().set_local(x.getArray()[:])
        self.rho.vector().apply('')
        dc=self.Jhat.derivative().vector().get_local()
        dv=self.Vhat.derivative().vector().get_local()
        return np.array([dc,dv])
    
    def optimize(self):
        # Set loglevel to 30 to disable some fenics messages...
        set_log_level(30)
        # For output of optimization history
        self.iter=0
        # Save markings for inspection in paraview
        File(self.directory+'boundary_parts.pvd') << self.mech.boundary_parts
        # Output files
        self.u_fid=File(self.directory+'u.pvd')
        self.rho_fid=File(self.directory+'rho.pvd')
        self.rho_tilde_fid=File(self.directory+'rho_tilde.pvd')
        self.rho_bar_fid=File(self.directory+'rho_bar.pvd')
        self.nout=5

        self.beta=1
        self.eta=0.5
        betaMax=64
        while self.beta<=betaMax:
            if self.rank==0:
                print('beta=',self.beta)
            self.setUpFunctionals(self.beta,self.eta)
            self.mma.solve( self.rho_petsc )
            self.beta=self.beta*2
        self.beta=self.beta/2
        self.plot_k(self.mma.x,True)
        (u,rho_bar,rho_tilde)=self.Forward(self.beta,self.eta)
        eps=self.mech.epsilon(u)
        sig=self.mech.sigma(u,eps)
        Energy=float(assemble(0.5*inner(sig,eps)*dx,annotate=False))
        Volume=float(assemble((Constant(1)+1e-9*rho_bar)*dx))
        UsedVolume=float(assemble(rho_bar*dx))
        return (Energy,UsedVolume,Volume)

if __name__ == "__main__":
    t=topOpt()
    if t.rank==0:
        print('DOFs in mechanics problem: ',t.mech.V.dim())
    (Energy,UsedVolume,Volume)=t.optimize()
    if t.rank==0:
        print('Final Average Strain Energy Density: ',Energy/Volume,'Final Volume fraction:',UsedVolume/Volume)
   
