"""
| Test for mma using numpy version of MMA.
| Run with
|   python3 test7pet.py
| or
|   mpirun -n 2 python3 test7pet.py
| or
|   mpirun -n 3 python3 test7pet.py
"""

from mma import MMA_petsc
from mma_tester import MMA_tester
from petsc4py import PETSc
import numpy as np

class test_problem7_pet():
    """
|   solves a min-max problem:
|     min max (x1,x2,x3) s.t. x1 + x2 + x3 = 15
    """
    def __init__(self,x0):
        # 5 constraints, since 3 min-max functions + 2 constraints
        self.mma=MMA_petsc(x0,5,f=self.f,g=self.g) 
        self.mma.xmin[:]=-100
        self.mma.xmax[:]=100
        self.mma.setMinMax(3,2) # 3 min-max functions and 2 constraints
        self.exact=np.array([5,5,5])

    def f(self,x):
        y=self.mma.parToLocal(x)
        # Keep min-max functions >0
        c=50
        delta=1e-8
        ar=np.array([ 0 , y[0]+c , y[1]+c , y[2]+c ,
                      y[0]+y[1]+y[2]-15-delta , -y[0]-y[1]-y[2]+15-delta ])
        return ar

    def g(self,x):
        ar=np.array([ [0,0,0] ,  [1,0,0] ,  [0,1,0] ,  [0,0,1] ,
                      [1,1,1] , [-1,-1,-1] ])
        return ar[:,self.mma.nlow:self.mma.nhigh]

if __name__ == "__main__":
    import sys
    np.set_printoptions(linewidth=180)
    
    rank=PETSc.Comm(PETSc.COMM_WORLD).getRank()
    
    if rank==0:
        print('Running test problem 7 using PETSc')
    x0=PETSc.Vec().createMPI(3)
    x0.setValues(range(0,3),[0,0,0])
    x0.assemble()
    t7p=test_problem7_pet(x0)

    # Test gradients with finite differences
    t7p.mma.testGrads(components=[0,1,2])
    
    sol7=t7p.mma.solve(x0)
    tester=MMA_tester(t7p.exact)
    test=tester.test(sol7)            # Test solution
    local_sol=t7p.mma.parToLocal(sol7)
    if rank==0:
        print('Solution',local_sol)
        print('PETSc test problem 7 done! Passed:',test)
        print()
