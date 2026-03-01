from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from ticToc import TicToc

list_linear_solver_methods()
list_krylov_solver_preconditioners()

nx=100
ny=nx
mesh = RectangleMesh.create([Point(-5, -5), Point(5,5)],[nx,ny],CellType.Type.triangle)
# or CellType.Type.quadrilateral but then plotting must then be done in paraview.

# Continuous function space
Vc=FunctionSpace(mesh,'Lagrange',1)
trial=TrialFunction(Vc)
test=TestFunction(Vc)

# Discontinuous function space of piecewise constant functions.
Vd=FunctionSpace(mesh,'DG',0)

class MyExpression(UserExpression):
    def eval(self, value, x):
        if (x[0]*x[0]+x[1]*x[1])<=3*3:
            # Value is 1 if inside circle of radius 3 centered in (0,0)
            value[0]=1.0
        else:
            # Value is 0 if outside circle
            value[0]=0.0
    def value_shape(self):
        return (1,)

# Create a FEniCS "DG0 function", from Vd, using MyExpression 
rho=interpolate( MyExpression(element=Vd.ufl_element()) ,Vd)

print('Minimum rho value: ',np.min(rho.vector()[:]))
print('Maximum rho value: ',np.max(rho.vector()[:]))

R=0.1 # Helmholtz filter radius

# Define weak form and let fenics choose the bilinear form (a) and linear form (L)
F=(inner(pow(R,2)*grad(trial),grad(test))+inner(trial,test))*dx - inner(rho,test)*dx
a=lhs(F)
L=rhs(F)

# Filtered rho
rho_tilde=Function(Vc)

print("DOFs in linear problem: ",Vc.dim())

# Solve the discrete equations with a direct solver and 
# store solution in rho_tilde
timer=TicToc(True)
timer.tic("Direct Solver")
solve(a == L,rho_tilde,solver_parameters={"linear_solver": "petsc"})
timer.toc("Direct Solver")

# Solve the discrete equations with an iterative solver and 
# overwrite the solution in rho_tilde
timer.tic("Iterative Solver")
solve(a == L,rho_tilde, solver_parameters={"linear_solver": "cg", "preconditioner": "hypre_amg"})
timer.toc("Iterative Solver")

print('Minimum rho tilde value: ',np.min(rho_tilde.vector()[:]))
print('Maximum rho tilde value: ',np.max(rho_tilde.vector()[:]))

# Plot input rho using FEniCS plot command
plt.figure(1)
p=plot(rho)
plt.title(r'$\rho$')
plt.colorbar(p)

# Plot input filtered rho
plt.figure(2)
p=plot(rho_tilde, mode='color')
plt.title(r'$\tilde{\rho}$ with R=%.1f'%R)
plt.colorbar(p)

# Define hyberbolic tangent which can be used in FEniCS calculations
def tanh(x):
    return (exp(2*x)-1)/(exp(2*x)+1)
# Define smoothed Heaviside function. Default tanh used is the one defined above
def Heavi(beta,eta,x,th=tanh):
    return (th(beta*eta)+th(beta*(x-eta)))/(th(beta*eta)+th(beta*(1-eta)))

# Plot smoothed Heaviside step function for different beta and eta
plt.figure(3)
x=np.linspace(0,1,100)

# Plot Smoothed Heaviside function.
# numpy's tanh is used, since x is a numpy vector.
plt.plot(x,Heavi(1,0.5,x,th=np.tanh),label=r"$\beta=1,\eta=0.5$")
plt.plot(x,Heavi(8,0.5,x,th=np.tanh),label=r"$\beta=8,\eta=0.5$")
plt.plot(x,Heavi(32,0.5,x,th=np.tanh),label=r"$\beta=32,\eta=0.5$")
plt.plot(x,Heavi(32,0.25,x,th=np.tanh),label=r"$\beta=32,\eta=0.25$")
plt.plot(x,Heavi(32,0.75,x,th=np.tanh),label=r"$\beta=32,\eta=0.75$")
plt.legend()

# Plot the "smoothed Heaviside projection" of the filtered rho.
# Default tanh is used, since it must work with FEniCS functions.
rho_bar1=project(Heavi(32,0.4,rho_tilde),Vd,solver_type="cg",preconditioner_type="hypre_amg")

plt.figure(4)
p=plot(rho_bar1)
plt.title(r'$\bar{\rho}$ with $\beta=32$ and $\eta=0.4$')
plt.colorbar(p)

rho_bar2=project(Heavi(32,0.6,rho_tilde),Vd,solver_type="cg",preconditioner_type="hypre_amg")

plt.figure(5)
p=plot(rho_bar2)
plt.title(r'$\bar{\rho}$ with $\beta=32$ and $\eta=0.6$')
plt.colorbar(p)

# Plot the edge of the geometry, calculated as a smoothed-projected circle
# using eta 0.6 (small circle) subtracted from a smoothed-projected circle 
# using eta 0.4 (big circle).
# Can be L2 projected to Vd or Vc, here Vc is used to give a "smooth" edge.
edge=project(rho_bar1-rho_bar2,Vc,solver_type="cg",preconditioner_type="hypre_amg")

plt.figure(6)
p=plot(edge)
plt.title(r'edge')
plt.colorbar(p)

# Project using smoothed Heaviside function, to make the edge more "0-1".
plt.figure(7)
p=plot(project(Heavi(32,0.2,edge),Vd,solver_type="cg",preconditioner_type="hypre_amg"))
plt.title(r'Projected edge')
plt.colorbar(p)

plt.show()

