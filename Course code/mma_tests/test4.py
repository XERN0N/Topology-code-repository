"""
| Test for mma using numpy version of MMA.
| Run with
|   python3 test4.py
"""

from mma import MMA_numpy
from mma_tester import MMA_tester
import numpy as np

class test_problem4_np():
    """
|   solves
|    min sum x_i^2 s.t. sum x_i = 0, (sum x_i)**2 = 0, 0<= x_i <= 2
|   solution x_i = 0
|   Since x_i>=0, equality constraints are implemented simply as inequality constraints (<=0).
    """
    def __init__(self,x0):
        # Create instance of MMA
        self.mma=MMA_numpy(x0,2,f=self.f,g=self.g)
        # Set lower bound of variables
        self.mma.xmin[:]=0.0
        # Set upper bound of variables
        self.mma.xmax[:]=2.0
        # Get number of variables from MMA
        self.n=self.mma.n
        # Store exact solution for comparison
        self.exact=np.zeros(x0.size)

    def f(self,x):
        # Define cost function and constraint functions
        return np.array([ x.dot(x)/self.n**2 , x.sum()/self.n , x.sum()**2/self.n**2 ])

    def g(self,x):
        # Return the Jacobian of f
        tmp=np.zeros(self.n)
        tmp[:]=2*x[:]
        tmp2=np.ones(self.n)
        tmp3=2*x.sum()*tmp2
        return np.array([ tmp/self.n**2 , tmp2/self.n , tmp3/self.n**2 ])

if __name__ == "__main__":
    np.set_printoptions(linewidth=180)
    print('Running test problem 4 using numpy on CPU 0')

    # Create initial guess as a vectore with 10000 1s.
    x0=np.ones(10000)
    
    # Create instance of problem class
    t4=test_problem4_np(x0)

    # Test gradients with finite differences
    t4.mma.testGrads()

    # Solve problem
    sol4=t4.mma.solve(x0)

    # Create instance of tester class
    tester=MMA_tester(t4.exact)

    # Print stuff
    print('Solution',sol4)
    print('numpy test problem 4 done! Passed:',tester.test(sol4))
    print()
