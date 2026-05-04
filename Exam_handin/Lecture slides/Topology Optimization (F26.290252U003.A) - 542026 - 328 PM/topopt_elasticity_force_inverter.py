from fenics import *
from fenics_adjoint import *
from force_inverter import ForceInverter
from filter_and_project import FilterAndProject
import numpy as np
from mma import MMA_petsc

class topOpt():
    def __init__(self):
        # MPI rank
        self.rank=MPI.rank(MPI.comm_world)

        # Directory for output
        self.directory='topopt_force_inverter/'

        # Some constants
        density = 1380     # Density, [kg/m^3] 
        L = 1              # Length, [m]
        W = L/2            # Width, [m]
        g = 9.81           # Gravitational acceleration, [m/s^2]
        density = 1380     # Density, [kg/m^3] 
        F = 1e6            # Total force on boundary, [N]
        self.E0 = 3e9      # Youngs modulus, [Pa]
        self.nu = 0.38     # Poissons ration, [1]
        mesh_size=W/50     # Mesh size
        self.Volfrac=0.25
        self.VolMax=L*W
        
        R=W/16/(2*np.sqrt(3))  # Filter Radius ~ W/16
        #R=2*mesh_size/(2*np.sqrt(3))  # Minimum filter Radius
        self.Emin=self.E0/1e6 # Minimum E
        self.pen=3            # SIMP penalization 
        
        self.mech=ForceInverter(L,W,self.E0,self.nu,F,0,mesh_size)
        
        self.FP=FilterAndProject(R,self.mech.mesh)
        self.rho=interpolate(Constant(0.5),self.FP.Vd)
        self.rho.rename('rho','rho')
        self.rho_petsc=as_backend_type(self.rho.vector()).vec()  # Pointer to rho's petsc data structure

        # Set up optimizer
        # Number of constraints
        self.ncon=1
        self.setUpOptimizer( self.rho_petsc )
        self.fixParts()

    def fixParts(self):
        def InsideFullRegion(x):
            # Return true if coordinates in x is inside a region with rho=1
            return self.mech.insideLoad(x) or self.mech.insideMove(x) or self.mech.insideClamped(x)

        # Get rho's DOF coordinates
        coor = self.FP.Vd.tabulate_dof_coordinates().reshape((-1, self.mech.dim))
        size=self.rho.vector().get_local().shape[0]
        i=0
        rho0=np.zeros(size)
        regions=np.zeros(size)
        for c in coor:
            if InsideFullRegion(c):
                self.mma.xmin[i]=1.0
                self.mma.xmax[i]=1.0
                rho0[i]=1.0
            else:
                rho0[i]=0.5
            i=i+1

        self.rho.vector().set_local(rho0)  # Put values into the rho vector
        self.rho.vector().apply('')        # Needed for parallel execution
        File(self.directory+'/InitialDeisgn.pvd') << self.rho

    def setUpOptimizer(self,x0):
        """Initializes MMA"""
        self.mma=MMA_petsc(x0,self.ncon,f=self.f,g=self.g,plot_k=self.plot_k)
        self.mma.xmin[:]=0.0
        self.mma.xmax[:]=1.0

        # MMA setup
        self.mma.move=0.5
        self.mma.xtol=0.001
        self.mma.ftol=0.0001
        self.mma.lmax=5
        self.mma.kmax=30
        self.mma.kmin=1
        self.mma.it=0
        self.mma.mma_timing=False  

    def Forward(self,beta,eta):
        rho_tilde=self.FP.filter(self.rho)
        rho_bar,_=self.FP.Heavi_project(beta,eta,rho_tilde)
        E=self.Emin+pow(rho_bar,self.pen)*(self.E0-self.Emin) # SIMP E
        self.mech.E=E
        self.mech.mask=rho_bar
        u=self.mech.solve()
        return (u,rho_bar,rho_tilde)

    def plot_k(self,x,force_plot=False):
        """Abstract method from mma, which can be used for output during optimization""" 
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
        # Make functionals
        m=Control(self.rho)
        J=assemble( u[0]/self.mech.MoveSize*self.mech.ds(4) )
        V=assemble( (rho_bar-self.Volfrac)/self.VolMax*dx )
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
        """Redefine function from MMA that calculates the vector function f"""
        self.rho.vector().set_local(x.getArray()[:])
        self.rho.vector().apply('')

        J=self.Jhat(self.rho)
        V=self.Vhat(self.rho)
        return np.array([J,V])

    def g(self,x):
        """Redefine function from MMA that calculates the gradient (Jacobian) of f"""
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
        Volume=float(assemble((Constant(1)+1e-20*rho_bar)*dx))
        J=float( assemble( u[0]/self.mech.MoveSize*self.mech.ds(4) ) )
        V=float( assemble( rho_bar*dx ) )/Volume
        return (J,V)
    
if __name__ == "__main__":
    t=topOpt()
    if t.rank==0:
        print('DOFs in mechanics problem: ',t.mech.V.dim())
    (J,V)=t.optimize()
    if t.rank==0:
        print('J',J)
        print('V',V)
   
