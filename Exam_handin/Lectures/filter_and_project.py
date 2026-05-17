from fenics import *
from fenics_adjoint import *

class FilterAndProject:
    def __init__(self,R,mesh):
        self.Vc=FunctionSpace(mesh,'Lagrange',1)
        self.Vd=FunctionSpace(mesh,'DG',0)
        self.R=float(R)
        
    def filter(self,rho,R=None):
        if R is not None:
            self.R=float(R)
        trial=TrialFunction(self.Vc)
        test=TestFunction(self.Vc)
        F=(inner(pow(self.R,2)*grad(trial),grad(test))+trial*test)*dx - rho*test*dx
        a=lhs(F)
        L=rhs(F)
        rho_tilde=Function(self.Vc,name='rho_tilde')
        solve(a == L, rho_tilde, solver_parameters={"linear_solver": "cg", "preconditioner": "hypre_amg"})
        return rho_tilde

    def tanh(self,x):
        # tanh(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x)) = (1-exp(-2*x))/(1+exp(-2*x)) = 2/(1+exp(-2*x))-1
        return 2/(1+exp(-2*x))-1
    def cosh(self,x):
        return (exp(x)+exp(-x))/2.0
    
    def Heavi(self,beta,eta,x,th=None):
        # Smoothed heaviside function
        if th is None:
            th=self.tanh
        return (th(beta*eta)+th(beta*(x-eta)))/(th(beta*eta)+th(beta*(1-eta)))
    def dHeavi(self,beta,eta,x):
        return beta/pow(self.cosh(beta*(x-eta)),2)/(self.tanh(beta*eta)+self.tanh(beta*(1-eta)))

    def Heavi2(self,beta,x):
        # Smoothed heaviside function with a step near x=0.
        return 1-exp(-beta*x)+x*exp(-beta)
    
    def project_Vd(self,u):
        # Calculates L2 projection of u onto function space Vd.
        trial=TrialFunction(self.Vd)
        test=TestFunction(self.Vd)
        a = inner(test, trial)*dx
        L = inner(test, u )*dx
        pu=Function(self.Vd,name='rho_bar')
        solve(a == L, pu, solver_parameters={"linear_solver": "cg", "preconditioner": "hypre_amg"})
        return pu

    def project_Vc(self,u):
        # Calculates L2 projection of u onto function space Vc.
        trial=TrialFunction(self.Vc)
        test=TestFunction(self.Vc)
        a = inner(test, trial)*dx
        L = inner(test, u )*dx
        pu=Function(self.Vc)
        solve(a == L, pu, solver_parameters={"linear_solver": "cg", "preconditioner": "hypre_amg"})
        return pu

    def Heavi_project(self,beta,eta,rho_tilde):
        # Projects rho_tilde to Vd and then projects H(u) to Vd.
        # Without this 2-stage projection, 1-element gradients from the functions in Vc
        # will create smoothed steps in rho_bar.
        u=self.project_Vd(rho_tilde)
        rho_bar=self.project_Vd(self.Heavi(beta,eta,u))
        return rho_bar,u

    def Heavi2_project(self,beta,eta,rho_tilde):
        # Projects rho_tilde to Vd and then projects H2(u) to Vd.
        u=self.project_Vd(rho_tilde)
        rho_bar=self.project_Vd(self.Heavi2(beta,u))
        return rho_bar,u

    def Heavi3_project(self,beta,eta,rho_tilde):
        # Projects rho_tilde to Vd and then projects H(beta/4,eta, H(beta,eta,u) ) to Vd.
        u=self.project_Vd(rho_tilde)
        rho_bar=self.project_Vd(self.Heavi(max(1.0,beta/4),0.5,self.Heavi(beta,eta,u)))
        return rho_bar,u

    def Hard_project(self,rho_tilde):
        # Projects rho_tilde to Vd and then projects u>=0.5 to Vd
        from ufl import ge,conditional
        u=self.project_Vd(rho_tilde)
        rho_bar=self.project_Vd( conditional(ge(u,0.5),1.0,0.0) )
        return rho_bar,u
    
    def filter_gradient(self,g,rho,R=None):
        # Manual adjoint gradient depending on rho_tilde(rho)
        if R is not None:
            self.R=float(R)
        trial=TrialFunction(self.Vc)
        test=TestFunction(self.Vc)
        Fv=(inner(pow(self.R,2)*grad(trial),grad(test))+trial*test)*dx - g*test*dx
        av=lhs(Fv)
        Lv=rhs(Fv)
        v=Function(self.Vc)
        solve(av == Lv, v, solver_parameters={"linear_solver": "cg", "preconditioner": "hypre_amg"})
        drho=TestFunction(self.Vd)
        J=derivative(rho,rho,drho)
        df0_adjoint=assemble(v*J*dx)
        return df0_adjoint
    
if __name__ == "__main__":
    # Run some tests if file is executed directly
    import numpy as np
    import matplotlib.pyplot as plt
    from pyadjoint.tape import annotate_tape
    while not annotate_tape():
        continue_annotation()  # Run pause until we actually pause annotating...

    class MyExpression(UserExpression):
        def eval(self, value, x):
            if (x[0]*x[0]+x[1]*x[1])<=3*3:
                value[0]=1.0
            else:
                value[0]=0.0
        def value_shape(self):
            return (1,)

    nx=10
    ny=nx
    mesh = RectangleMesh.create([Point(-5, -5), Point(5,5)],[nx,ny],CellType.Type.triangle)

    R=0.1
    FP=FilterAndProject(R,mesh)
    rho_wanted=interpolate( MyExpression(element=FP.Vc.ufl_element()) ,FP.Vd)
    rho=interpolate( Expression('abs(sin(x[0]+x[1]))', degree=1) ,FP.Vd)

    rho_tilde=FP.filter(rho)
    
    # Cost function
    beta=4
    eta=0.25
    
    rho_bar,_=FP.Heavi_project(beta,eta,rho_tilde)
    cost=(rho_wanted-rho_bar)**2

    m=Control(rho)
    f=assemble( cost*dx )
    fHat=ReducedFunctional(f,m)

    f0=fHat(rho)
    print(f0)
    df0=fHat.derivative().vector().get_local()
    print(df0[0:10])

    while annotate_tape():
        pause_annotation()  # Run pause until we actually pause annotating...

    rho_bar,u=FP.Heavi_project(beta,eta,rho_tilde)
    cost=(rho_wanted-rho_bar)**2
    dcost=-2*(rho_wanted-rho_bar)*FP.dHeavi(beta,eta,u)
    df0_adjoint=FP.filter_gradient(dcost,rho)
    
    print(df0_adjoint.get_local()[0:10])
    print()
    print( np.abs(df0-df0_adjoint.get_local())[0:10] )

    print( np.max( np.abs(df0-df0_adjoint.get_local()) ) )

