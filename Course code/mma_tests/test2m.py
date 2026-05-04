"""
| Test for mma using numpy version of MMA.
| Run with
|   python3 test2m.py
"""

from mma import MMA_numpy
from mma_tester import MMA_tester
import numpy as np

class test_problem2m_np():
    """
|   solves
|     min 2x[0]-x[0]**3 s.t. x[0]<=6, x[0]>=0, -10<=x<=10
|   solution x[0]=0
    """
    def __init__(self,x0):
        self.mma=MMA_numpy(x0,2,f=self.f,g=self.g)
        self.mma.xmin[:]=-10
        self.mma.xmax[:]=10
        self.exact=np.array([0.0])

    def f(self,x):
        return np.array([ 2*x[0]-x[0]**3 , x[0]-6 , -x[0] ])

    def g(self,x):
        return np.array([ [2-3*x[0]**2] , [1] , [-1] ])

if __name__ == "__main__":
    np.set_printoptions(linewidth=180)
    x0=np.array([0.3])
    t2m=test_problem2m_np(x0)
    tester=MMA_tester(t2m.exact)
    sol2m=t2m.mma.solve(x0)
    print('Solution',sol2m)
    print('numpy test problem 2m done. Passed:',tester.test(sol2m))
    print()
