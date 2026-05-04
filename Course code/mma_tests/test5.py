"""
| Test for mma using numpy version of MMA.
| Run with
|   python3 test5.py
"""

from mma import MMA_numpy
from mma_tester import MMA_tester
import numpy as np

class test_problem5_np():
    """
|   solves
|     min 2x[0]-x[0]**3+3x[1]-2[x]**2 s.t. x[0]+3x[1]<=6, 6x[0]+2x[1]<=10, x[0]+x[1]=2
|   solutions [[0, 2], [2, 0]]
|   
|   Equality constraint implemented as two inequalities, x[0]+x[1]<=2 and -x[0]-x[1]<=-2  (i.e. x[0]+x[1]>=2)
    """
    def __init__(self,x0):
        self.mma=MMA_numpy(x0,4,f=self.f,g=self.g)
        self.mma.xmin[:]=-10
        self.mma.xmax[:]=10
        self.exact=np.array([ [0, 2] , [2, 0] ])

    def f(self,x):
        return np.array([ 2*x[0]-x[0]**3+3*x[1]-2*x[1]**2 , x[0]+3*x[1]-6 , 5*x[0]+2*x[1]-10 , x[0]+x[1]-2 , -x[0]-x[1]+2 ])

    def g(self,x):
        return np.array([ [2-3*x[0]**2,3-4*x[1]] , [1,3] , [5,2] , [1,1] , [-1,-1] ])

if __name__ == "__main__":
    np.set_printoptions(linewidth=180)
        
    x0=np.array([1.5,0.5])
    t5=test_problem5_np(x0)
    tester=MMA_tester(t5.exact)
    sol5=t5.mma.solve(x0)
    print('Solution',sol5)
    print('numpy test problem 5 done! Passed:',tester.test(sol5))
    print()
    print('Running test problem 5 part 2 (different initial guess)')
    x0=np.array([0.5,1.5])
    sol5_2=t5.mma.solve(x0)
    print('Solution',sol5_2)
    print('numpy test problem 5 part 2 done! Passed:',tester.test(sol5_2))
    print()
 
