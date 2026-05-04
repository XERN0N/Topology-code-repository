# Run in serial with:
#  python3 Ex6.py
#
# Run in parallel on 2 cores with:
#  mpirun -n 2 python3 Ex6.py
#
# If performance seems poor, try the following commands and see if it improves:
#  OMP_NUM_THREADS=1 python3 Ex6.py
#  OMP_NUM_THREADS=1 mpirun -n 2 python3 Ex6.py

from mma import MMA_petsc, MMA_numpy
from petsc4py import PETSc
import numpy as np
from random import seed,random
import matplotlib.pyplot as plt
from ticToc import TicToc

# seed random number generator to always generate the same numbers
seed(1)

# Get the CPU number this instance runs on.
rank=PETSc.Comm(PETSc.COMM_WORLD).getRank()

if rank==0:
    # This part should only run on CPU 0
    class q1_np():
        # min (x[0]**2-1)**2+cos(4x[1]) s.t. x[0]+x[1] <= 0
        def __init__(self,x0,lb,ub):
            self.mma=MMA_numpy(x0,1,f=self.f,g=self.g)
            self.mma.xmin[:]=lb
            self.mma.xmax[:]=ub
            self.resetPoints()

        def resetPoints(self):
            # Clear recorded points
            self.points=None

        def addPoint(self,x):
            # Add point to list of points
            if self.points is None:
                self.points=np.array(x[:])
            else:
                self.points=np.vstack([self.points, x])

        def f(self,x):
            return np.array([ (x[0]**2-1)**2+np.cos(4*x[1]) ,
                              x[0]+x[1] ])

        def g(self,x):
            self.addPoint(x) # Record where gradients are evaluated
            return np.array([ [2*(x[0]**2-1)*2*x[0],-4*np.sin(4*x[1])] ,
                              [1,1] ])

    # Number of solutions to run and plot
    nsol=10
    # From -size/2 to size/2 in x and from -size to size in y
    size=5
    x0=np.array([0,0])
    q1=q1_np(x0,[-size/2,-size],[size/2,size])
    q1.mma.lmax=5 # Try to play with this, from 1 to 10.
    q1.mma.output=False # Turn off MMA iterations output

    def f(x1,x2):
        return (x1**2-1)**2+np.cos(4*x2)

    x=np.linspace(-size/2,size/2,1000)
    y=np.linspace(-size,size,1000)
    X,Y=np.meshgrid(x,y)
    plt.figure(1)
    # Plot cost function
    #plt.pcolormesh(x, y, f(X,Y))
    plt.contourf(x, y, f(X,Y), levels=50)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    cbar=plt.colorbar()
    cbar.ax.set_title('f')
    # Plot constraint boundary line
    plt.plot(x,-x, linewidth='2.5', c='#000000')
    
    for i in range(0,nsol):
        # Generate random starting point
        x0[0] = 2*size/2*(0.5-random())
        x0[1] = 2*size*(0.5-random())
        # Reset recorded points
        q1.resetPoints()
        # Solve using MMA
        s=q1.mma.solve(x0)
        # plot line with o's through recorded points
        plt.plot(q1.points[:,0],q1.points[:,1],'--x', linewidth=1,markersize=5, zorder=1)
        # Plot small x at inital point big x at solution, use zorder to plot on top of o-line
        plt.scatter(x0[0],x0[1],25, marker='o', zorder=2, c='#ffffff')
        plt.scatter(s[0],s[1],250, marker='x', zorder=3, c='#ff0000')
    plt.title('MMA iterations. Feasible region is below thick black line.\nWhite dot: Starting point, dashed line: Iterations,\n red X: Solution found by MMA')

# This part will be run on all CPUs
class q2_petsc():
    # min (x**2-1)**2 s.t. sum x <= sqrt(n) and sum x >=-sqrt(n)
    #
    def __init__(self,x0):
        self.mma=MMA_petsc(x0,2,f=self.f,g=self.g)
        self.mma.xmin[:]=-10
        self.mma.xmax[:]=10
        # self.mma.n is number of variables in problem
        # self.mma.nlocal is number of variables on the CPU

    def f(self,x):
        # Scale cost function with n**2 and constraints n to get 'reasonable' values for MMA.
        return np.array([ (x.dot(x)-1)**2/self.mma.n**2 , (x.sum()-np.sqrt(self.mma.n))/self.mma.n , -(x.sum()-np.sqrt(self.mma.n))/self.mma.n ])
        #return np.array([ (x.dot(x)-1)**2 , (x.sum()-np.sqrt(self.mma.n)) , -(x.sum()-np.sqrt(self.mma.n)) ]) # No scaling used, poor performance for large n

    def g(self,x):
        ddot=np.zeros(self.mma.nlocal)
        ddot[:]=4*(x.dot(x)-1)*x.getArray()[:]
        dsum=np.ones(self.mma.nlocal)
        return np.array([ ddot/self.mma.n**2 , dsum/self.mma.n , -dsum/self.mma.n ])
        #return np.array([ ddot , dsum , -dsum ]) # No scaling used

 
timer=TicToc(rank==0) # Print timing only on CPU 0
nvars=np.array([10, 100, 1000, 10000, 100000]) # Use this line when running to find the timing
#nvars=np.array([10, 100])                       # Use this line to view the graphs of previous timings quickly
times=np.zeros(nvars.size)
for i in range(0,nvars.size):
    n=nvars[i]
    x0=PETSc.Vec().createMPI(n)
    x0.setValues(range(0,n),np.ones(n))
    x0.assemble()

    q2=q2_petsc(x0)
    q2.mma.output=False # No output from MMA. Output can mess up detailed timing.

    #q2.mma.testGrads() # testGrads is useful for finding mistakes in gradient calculations...
    
    timer.tic("MMA solve")
    sol=q2.mma.solve(x0)
    times[i]=timer.toc("MMA solve")

    d=sol.dot(sol)
    s=sol.sum()

    epsilon=1e-3
    if rank==0:
        print('n=%i'%n, 'x^2=%.5f'%d , 'sum x=%.5f'%s , 'sqrt(n)=%.5f'%np.sqrt(n) ,
              'Constraints satisfied within %g: '%epsilon, s<=(np.sqrt(n)+epsilon) , s>=(-np.sqrt(n)-epsilon) )
        print()

  
if rank==0:
    print(times)

    # Plot times from previous runs on my 4-core laptop.
    nvars=np.array([ 10, 100, 1000, 10000, 100000])
    # Results when not scaling the functions supplied to MMA
    #cpu1=np.array([0.0721640587, 0.0887069702, 0.102148533, 3.64917111, 167.346086])
    #cpu2=np.array([ 0.08545828,  0.09034753,  0.10761046,  2.88463688, 49.52625442])
    #cpu3=np.array([ 0.11988521, 0.16573787, 0.19667149, 4.00768518, 59.06564116])
    #cpu4=np.array([ 0.17724109, 0.21616054, 0.24403691, 4.47770762, 44.09315729])
    # Results when scaling the functions supplied to MMA
    cpu1=np.array([0.06237674, 0.06380606, 0.06414151, 0.14804673, 1.30263042])
    cpu2=np.array([0.07208014, 0.0744648,  0.07915044, 0.1211741,  0.91634774])
    cpu3=np.array([0.08003831, 0.07875538, 0.08292699, 0.11061406, 0.72134757])
    cpu4=np.array([0.08694553, 0.09937978, 0.11294317, 0.11845922, 0.62869573])

    plt.figure()
    plt.title("Results from previous runs on a 4-core laptop.")
    plt.plot(nvars,cpu1,'*-',label="1 CPU")
    plt.plot(nvars,cpu2,'*-',label="2 CPU")
    plt.plot(nvars,cpu3,'*-',label="3 CPU")
    plt.plot(nvars,cpu4,'*-',label="4 CPU")
    plt.xlabel("n")
    plt.ylabel("t [s]")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()

    plt.figure()
    plt.title("Results from previous runs on a 4-core laptop.")
    plt.plot(nvars,cpu1/cpu1,'*-',label="1 CPU")
    plt.plot(nvars,cpu1/cpu2,'*-',label="2 CPU")
    plt.plot(nvars,cpu1/cpu3,'*-',label="3 CPU")
    plt.plot(nvars,cpu1/cpu4,'*-',label="4 CPU")
    plt.xlabel("n")
    plt.ylabel("Speed-up wrt. 1 CPU")
    plt.xscale('log')
    plt.legend()
    plt.show()
