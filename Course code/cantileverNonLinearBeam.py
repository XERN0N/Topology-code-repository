from fenics import *
from cantileverLinearBeam import CantileverLinearBeam

class CantileverNonLinearBeam(CantileverLinearBeam):
    # Inherit everything from CantileverLinearBeam

    # Redefine epsilon
    def epsilon(self,u):
        # Green-Lagrange strain
        return (grad(u)+grad(u).T+dot(grad(u),grad(u).T))/2
    
    # Redefine solve
    def solve(self,initial_guess=None):
        if initial_guess is None:
            # Create zero vector is no initial_guess is provided
            u = Function(self.V,name='Displacement')
        else:
            # Otherwise use initial_guess as a starting point,
            # also to store solution
            u=initial_guess

        # Define variational problem
        du = TrialFunction(self.V)
        v = TestFunction(self.V)
        f = Constant((0, 0, 0))      # No body force
        T = Constant((0, 0, self.traction))

        eps = self.epsilon(u)

        # Stored strain energy density
        Psi = self.mu*tr(eps*eps)+self.lambda_/2*tr(eps)**2

        # Total potential energy
        Pi = Psi*dx - dot(f, u)*dx - dot(T, u)*self.ds(1)

        # Directional derivative about u in the direction of v
        F = derivative(Pi, u, v)
        # Compute Jacobian of F
        dF = derivative(F, u, du)

        # Create nonlinear variational problem and solve
        problem = NonlinearVariationalProblem(F, u, bcs=self.bc, J=dF)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters['newton_solver']['absolute_tolerance'] = 1e-6
        solver.parameters['newton_solver']['linear_solver'] = 'cg'
        solver.parameters['newton_solver']['preconditioner'] = 'hypre_amg'
        solver.solve()
        return u

if __name__ == "__main__":
    # This part will be executed if the file is run as
    # python3 cantileverdNonLinearBeam.py
    #
    # Calculate magnitude of displacement
    beam=CantileverNonLinearBeam(0.025,3e4)
    u=beam.solve()
    von_Mises=beam.vonMises(u)
    u_magnitude=beam.norm_u(u)
    print('min/max u:',
          u_magnitude.vector().min(),
          u_magnitude.vector().max())
    # Save solution to file in VTK format
    File('cantileverNonLinearBeam/displacement.pvd') << u
    File('cantileverNonLinearBeam/von_mises.pvd') << von_Mises
    File('cantileverNonLinearBeam/magnitude.pvd') << u_magnitude
