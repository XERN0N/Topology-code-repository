"""
| Test for mma using numpy version of MMA.
| Run with
|   python3 test3pet.py
| or
|   mpirun -n 2 python3 test3pet.py
"""

from mma import MMA_petsc
from mma_tester import MMA_tester
from petsc4py import PETSc
import numpy as np

class test_problem3_pet():
    """
|   solves
|     min 2x[0]-x[0]**3+3x[1]-2[x]**2 s.t. x[0]+3x[1]<=6, 6x[0]+2x[1]<=10, x[0]>=0,x[1]>=0
|   solutions [[1.38462, 1.53846], [0, 2], [2, 0]]
|    
|   Lower bounds implemented as general constraints
    """
    def __init__(self,x0):
        self.mma=MMA_petsc(x0,4,f=self.f,g=self.g)
        self.mma.xmin[:]=-10
        self.mma.xmax[:]=10
        self.exact=np.array([ [1.38462,1.53846] , [0,2] , [2,0] ])

    def f(self,x):
        y=self.mma.parToLocal(x)
        ar=np.array([ 2*y[0]-y[0]**3+3*y[1]-2*y[1]**2 , y[0]+3*y[1]-6 , 5*y[0]+2*y[1]-10 , -y[0] , -y[1] ])
        return ar

    def g(self,x):
        y=self.mma.parToLocal(x)
        ar=np.array([ [2-3*y[0]**2,3-4*y[1]] , [1,3] , [5,2] , [-1,0] , [0,-1] ])
        return ar[:,self.mma.nlow:self.mma.nhigh]


if __name__ == "__main__":
    import sys
    np.set_printoptions(linewidth=180)
    
    rank=PETSc.Comm(PETSc.COMM_WORLD).getRank()

    if rank==0:
        print('Running test problem 3 using PETSc')
        print('This problem has non-linear cost function and 4 linear constraint functions')
    x0=PETSc.Vec().createMPI(2)
    x0.setValues(range(0,2),[1.5,0.5])
    x0.assemble()
    t3p=test_problem3_pet(x0)

    # Test gradients with the Taylor test
    t3p.mma.testGradsTaylor()
    # Test gradients with finite differences
    t3p.mma.testGradsTaylor()

    sol3=t3p.mma.solve(x0)
    tester=MMA_tester(t3p.exact)
    test=tester.test(sol3)            # Test solution
    local_sol=t3p.mma.parToLocal(sol3) # Put whole solution vector in an array on each CPU
    if rank==0:
        print('Solution',local_sol)
        print('PETSc test problem 3 done! Passed:',test)
        print()
