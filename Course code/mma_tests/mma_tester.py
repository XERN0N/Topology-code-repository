from mma import MMA_petsc
from petsc4py import PETSc
import numpy as np

class MMA_tester():
    # Class to compare if a solution is almost equal to one of
    # the exact solution in self.exact.
    def __init__(self,exact=None):
        if exact is not None:
            self.exact=exact
    
    def near(self,a,b):
        return np.amax(np.absolute(a-b))<1e-4

    def test(self,sol):
        # Tests sol against self.exact
        if isinstance(sol,PETSc.Vec):
            sol=MMA_petsc.parToLocal(sol)
        p=False
        if len(self.exact.shape)==1:
            p=self.near(self.exact[:],sol)
        else:
            nsol=self.exact.shape[0]
            for i in range(0,nsol):
                p=p or self.near(self.exact[i,:],sol)
        return p
