"""
| Test for mma using numpy version of MMA.
| Run with
|   python3 test7.py
"""

from mma import MMA_numpy
from mma_tester import MMA_tester
import numpy as np

class test_problem7_np():
    """
|   solves a min-max problem:
|     min max (x1,x2,x3) s.t. x1 + x2 + x3 = 15
    """
    def __init__(self,x0):
        self.mma=MMA_numpy(x0,5,f=self.f,g=self.g) # 5 = 3 min-max functions + 2 constraints
        self.mma.xmin[:]=-100
        self.mma.xmax[:]=100
        self.mma.setMinMax(3,2) # 3 min-max functions and 2 constraints
        self.exact=np.array([5,5,5])

    def f(self,x):
        # Keep min-max functions >0
        c=50
        delta=1e-8
        ar=np.array([ 0 , x[0]+c , x[1]+c , x[2]+c , x[0]+x[1]+x[2]-15-delta , -x[0]-x[1]-x[2]+15-delta ])
        return ar

    def g(self,x):
        ar=np.array([ [0,0,0] ,  [1,0,0] ,  [0,1,0] ,  [0,0,1] , [1,1,1] , [-1,-1,-1] ])
        return ar

if __name__ == "__main__":
    np.set_printoptions(linewidth=180)

    print('Running test problem 7 using numpy on CPU 0')
    x0=np.array([0,0,0])
    t7=test_problem7_np(x0)
    
    # Test gradients with finite differences
    t7.mma.testGrads(components=[0,1,2])
    
    sol7=t7.mma.solve(x0)
    tester=MMA_tester(t7.exact)
    print('Solution',sol7)
    print('numpy test problem 7 done! Passed: ',tester.test(sol7))
    print()
