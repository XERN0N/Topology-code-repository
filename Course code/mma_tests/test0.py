"""
| Test for mma using numpy version of MMA.
| Run with
|   python3 test0.py
"""
from mma import MMA_numpy
from mma_tester import MMA_tester
import numpy as np

class test_problem0_np():
    """
|   solves
|     min x[0]**2+x[1]**2 s.t. 0<=x<=2
|   Solution x[0]=0, x[1]=0
|   At least one constraint is needed, so x[0]<=1000 is included.
    """
    def __init__(self,x0):
        # Create an instance of mma
        self.mma=MMA_numpy(x0,1,f=self.f,g=self.g)
        # Define lower bound of variables
        self.mma.xmin[:]=0
        # Define upper bound of variables
        self.mma.xmax[:]=2
        # Store exact solution, to be used for testing the solution.
        # In real problems, exact solution is of course not known.
        self.exact=np.array([0,0])
        # Use tighter tolerances than MMA default.
        self.mma.xtol=1e-4
        self.mma.ftol=1e-4

    def f(self,x):
        # Functions used to define the problem.
        return np.array([ x[0]**2+x[1]**2 , x[0]-1000 ])

    def g(self,x):
        # Return array of gradients of the function in f (the Jacobian of f).
        return np.array([ [2*x[0],2*x[1]] , [1,0] ])


if __name__ == "__main__":
    np.set_printoptions(linewidth=180)

    # Define initial guess
    x0=np.array([1,2])
    # Create instance of the problem to solve
    t=test_problem0_np(x0)
    # Solve problem using mma
    sol0=t.mma.solve(x0)
    # Create instance of tester class to test numerical against exact solution.
    tester=MMA_tester(t.exact)
    # Print information
    print('Solution',sol0)
    print('numpy test problem 0 done! Passed: ',tester.test(sol0))
    print()
