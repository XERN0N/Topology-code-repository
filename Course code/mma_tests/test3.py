"""
| Test for mma using numpy version of MMA.
| Run with
|   python3 test3.py
"""

from mma import MMA_numpy
from mma_tester import MMA_tester
import numpy as np

class test_problem3_np():
    """
|   solves
|    min 2x[0]-x[0]**3+3x[1]-2[x]**2 s.t. x[0]+3x[1]<=6, 6x[0]+2x[1]<=10, x[0]>=0,x[1]>=0
|   solutions [[1.38462, 1.53846], [0, 2], [2, 0]]
|   Lower bounds implemented as general constraints
    """
    def __init__(self,x0):
        self.mma=MMA_numpy(x0,4,f=self.f,g=self.g)
        self.mma.xmin[:]=-10
        self.mma.xmax[:]=10
        self.exact=np.array([ [1.38462,1.53846] , [0,2] , [2,0]])

    def f(self,x):
        return np.array([ 2*x[0]-x[0]**3+3*x[1]-2*x[1]**2 , x[0]+3*x[1]-6 , 5*x[0]+2*x[1]-10 , -x[0] , -x[1]])

    def g(self,x):
        return np.array([ [2-3*x[0]**2,3-4*x[1]] , [1,3] , [5,2] , [-1,0] , [0,-1] ])

if __name__ == "__main__":
    np.set_printoptions(linewidth=180)
    x0=np.array([1.5,0.5])
    t3=test_problem3_np(x0)
    tester=MMA_tester(t3.exact)
    sol3=t3.mma.solve(x0)
    print('Solution',sol3)
    print('numpy test problem 3 done! Passed:',tester.test(sol3))
    print()
