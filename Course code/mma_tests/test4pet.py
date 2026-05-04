"""
| Test for mma using numpy version of MMA.
| Run with
|   python3 test4pet.py
| or
|   mpirun -n N python3 test4pet.py
| with N replaced by the number of (physical) cores to use.
"""

from mma import MMA_petsc
from mma_tester import MMA_tester
from petsc4py import PETSc
import numpy as np

class test_problem4_pet():
    """
|   min sum x_i^2 s.t. sum x_i = 0, (sum x_i)**2 = 0, 0<= x_i <= 2
|   solution x_i = 0
|   Since x_i>=0, equality constraints are implemented simply as inequality constraints (<=0).
    """
    def __init__(self,x0):
        # Create instance of MMA
        self.mma=MMA_petsc(x0,2,f=self.f,g=self.g)
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
        # Return part of the Jacobian of f stored on the CPU core
        tmp=np.zeros(self.mma.nlocal)
        tmp[:]=2*x.getArray()[:]
        tmp2=np.ones(self.mma.nlocal)
        tmp3=2*x.sum()*tmp2
        return np.array([ tmp/self.n**2 , tmp2/self.n , tmp3/self.n**2 ])

if __name__ == "__main__":
    import sys
    np.set_printoptions(linewidth=180)

    # Get rank of CPU core the code is running on
    rank=PETSc.Comm(PETSc.COMM_WORLD).getRank()

    # Print stuff only on CPU core 0
    if rank==0:
        print('Running test problem 4 using PETSc')

    # Create initial guess as a PETSc vectore with 10000 1s.
    x0=PETSc.Vec().createMPI(10000)
    x0.setValues(range(0,10000),np.ones(10000))
    x0.assemble()

    # Create instance of problem class
    t4p=test_problem4_pet(x0)

    # Test gradients with the Taylor test
    if rank==0:
        print('Note that cost function in non-linear, 1st constraint is linear and 2nd constraint is non-linear')
    t4p.mma.testGradsTaylor()
    # Test gradients with finite differences
    t4p.mma.testGrads()
    
    # Solve problem
    sol4p=t4p.mma.solve(x0)

    # Create instance of tester class
    tester=MMA_tester(t4p.exact)

    # Test solution
    test=tester.test(sol4p)

    # Transfer solution so it is present on all CPU cores
    local_sol=t4p.mma.parToLocal(sol4p)

    # Print stuff only on CPU core 0
    if rank==0:
        print('Solution',local_sol)
        print('PETSc test problem 4 done! Passed:',test)
        print()
