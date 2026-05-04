"""
| Test for mma using numpy version of MMA.
| Run with
|   python3 test1.py
"""

from mma import MMA_numpy
from mma_tester import MMA_tester
import numpy as np

class test_problem1_np():
    """
|   solves
|     min x[0]**2+x[1]**2 s.t. -2[x0]+1 <=0, -3x[1]+2<=0, 0<=x<=2
|   Solution x[0]=1/2, x[1]=2/3
    """
    def __init__(self,x0):
        self.mma=MMA_numpy(x0,2,f=self.f,g=self.g)
        self.mma.xmin[:]=0
        self.mma.xmax[:]=2
        self.exact=np.array([0.5,2/3])

    def f(self,x):
        return np.array([ x[0]**2+x[1]**2 , -2*x[0]+1 , -3*x[1]+2 ])

    def g(self,x):
        return np.array([ [2*x[0],2*x[1]] , [-2,0] , [0,-3]])

if __name__ == "__main__":
    np.set_printoptions(linewidth=180)
    x0=np.array([1,2])
    t=test_problem1_np(x0)
    tester=MMA_tester(t.exact)
    sol1=t.mma.solve(x0)
    print('Solution',sol1)
    print('numpy test problem 1 done! Passed: ',tester.test(sol1))
    print()
