"""
| Test for mma using numpy version of MMA.
| Run with
|   python3 test5pet.py
| or
|   mpirun -n 2 python3 test5pet.py
"""

from mma import MMA_petsc
from mma_tester import MMA_tester
from petsc4py import PETSc
import numpy as np

class test_problem5_pet():
    """
|   solves
|    min 2x[0]-x[0]**3+3x[1]-2[x]**2 s.t. x[0]+3x[1]<=6, 6x[0]+2x[1]<=10, x[0]+x[1]=2
|   solutions [[0, 2], [2, 0]]
|   
|   Equality constraint implemented as two inequalities, x[0]+x[1]<=2 and -x[0]-x[1]<=-2  (i.e. x[0]+x[1]>=2)
    """
    def __init__(self,x0):
        self.mma=MMA_petsc(x0,4,f=self.f,g=self.g)
        self.mma.xmin[:]=-10
        self.mma.xmax[:]=10
        self.exact=np.array([ [0,2] , [2,0] ])

    def f(self,x):
        y=self.mma.parToLocal(x)
        ar=np.array([ 2*y[0]-y[0]**3+3*y[1]-2*y[1]**2 , y[0]+3*y[1]-6 , 5*y[0]+2*y[1]-10 , y[0]+y[1]-2 , -y[0]-y[1]+2 ])
        return ar

    def g(self,x):
        y=self.mma.parToLocal(x)
        ar=np.array([ [2-3*y[0]**2,3-4*y[1]] , [1,3] , [5,2] , [1,1] , [-1,-1] ])
        return ar[:,self.mma.nlow:self.mma.nhigh]

if __name__ == "__main__":
    import sys
    np.set_printoptions(linewidth=180)
    
    rank=PETSc.Comm(PETSc.COMM_WORLD).getRank()
    
    if rank==0:
        print('Running test problem 5 using PETSc')
    x0=PETSc.Vec().createMPI(2)
    x0.setValues(range(0,2),[1.5,0.5])
    x0.assemble()
    t5p=test_problem5_pet(x0)
    sol5=t5p.mma.solve(x0)
    tester=MMA_tester(t5p.exact)
    test=tester.test(sol5)            # Test solution
    local_sol=t5p.mma.parToLocal(sol5)
    if rank==0:
        print('Solution',local_sol)
        print('PETSc test problem 5 done! Passed:',test)
        print()

    if rank==0:
        print('Running test problem 5 using PETSc part 2 (different initial guess)')
    x0.setValues(range(0,2),[0.5,1.5])
    x0.assemble()
    sol52=t5p.mma.solve(x0)
    test=tester.test(sol52)            # Test solution
    local_sol=t5p.mma.parToLocal(sol52)
    if rank==0:
        print('Solution',local_sol)
        print('PETSc test problem 5 part 2 done! Passed:',test)
        print()
