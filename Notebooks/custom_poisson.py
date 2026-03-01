from fenics import UnitSquareMesh, FunctionSpace, DirichletBC, Expression, TrialFunction, TestFunction, dot, grad, dx, Function, solve, File, errornorm, Constant, plot
import matplotlib.pyplot as plt
import numpy as np

sq_mesh = UnitSquareMesh(64, 64)
V = FunctionSpace(sq_mesh, 'P', 1)

u_dirichlet = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

def boundary(x, on_boundary):
    return on_boundary

boundary_cond = DirichletBC(V, u_dirichlet, boundary)

#Variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = dot(grad(u), grad(v))*dx
L = f*v*dx

#get displacement field
u = Function(V)
solve(a == L, u, boundary_cond)

plot(sq_mesh)
p=plot(u)
plt.colorbar(p)

vtkfile = File('poisson_custom/solution.pvd')
vtkfile << u

error_l2 = errornorm(u_dirichlet, u, 'L2')

vertex_val_u_dirichlet = u_dirichlet.compute_vertex_values(sq_mesh)
vertex_val_u = u.compute_vertex_values(sq_mesh)

error_max = np.max(np.abs(vertex_val_u_dirichlet-vertex_val_u))

print('error L2 =', error_l2)
print('error max =', error_max)

plt.show()