import fenics as fs
import matplotlib.pyplot as plt
import numpy as np

#fs.list_krylov_solver_methods()
#fs.list_krylov_solver_preconditioners()
#fs.list_linear_algebra_backends()
#fs.list_linear_solver_methods()


nx, ny = 10, 10

sq_mesh = fs.RectangleMesh.create([
    fs.Point(-5, -5), fs.Point(5,5)],
    [nx, ny], 
    fs.CellType.Type.triangle
    )

f_space_continuous = fs.FunctionSpace(sq_mesh, 'Lagrange', 1)
f_space_discrete = fs.FunctionSpace(sq_mesh, 'DG', 0)

u_trial = fs.TrialFunction(f_space_continuous)
v_test = fs.TestFunction(f_space_continuous)


class input_function_disc(fs.UserExpression):
    """
    This class defines an input function of a disc.
    """
    def eval(self, value, x):
        """
        Evaluate whether or not x is on the disc and modify value to 1 or 0
        """
        if x[0]**2 + x[1]**2 <= 9:
            value[0]=1.0
        else:
            value[0]=0.0
    
    def value_shape(self):
        """
        Returns the shape of value
        """
        return (1,)
    
#Interpolation space rho in equations
rho = fs.interpolate(input_function_disc(element=f_space_discrete.ufl_element()), f_space_discrete)

print("minimum rho value: ", np.min(rho.vector()[:]))
print("maximum rho value: ", np.max(rho.vector()[:]))

Helmholz_radius = 0.1



print("end")


