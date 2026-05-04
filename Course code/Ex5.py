# Figures may look wrong in Spyder. Run in console with
# python3 Ex5.py

import numpy as np
import matplotlib.pyplot as plt

# Exercise 5.1
def f(x1,x2):
    return (x1**2-1)**2+np.cos(4*x2)

def df(x1,x2):
    return np.array([4*(x1**2-1)*x1,-4*np.sin(4*x2)])

x=np.linspace(-2,2,1000)
y=np.linspace(-4,4,1000)
X,Y=np.meshgrid(x,y)
plt.figure(1)
plt.pcolormesh(x, y, f(X,Y))
plt.xlabel('x_1')
plt.ylabel('x_2')
cbar=plt.colorbar()
cbar.ax.set_title('f')
#
# f=(x1**2-1)**2+cos(4*x2)
#
# solving grad f=0:
# 4*(x1**2-1)*x1=0 and -4sin(4*x2)=0 =>
# x1=+/- 1 or x1=0   and 4*x2=n pi, n is an integer
#
# The Hessian is:
# [4(x1**2-1)+8*x1**2             0 ]
# [0                  -16*cos(4*x2) ]
# 
# At the candidate optimum points the Hessian is:
#  When x1=+/- 1 and 4*x2=n pi:
#  [8             0 ]
#  [0 -16*cos(4*x2) ]
#  which is positive definite when cos(4*x2)<0 => n is odd
#  and indefinite when cos(4*x2)>0 => when n is even.
#
#  When x1=0 and 4*x2=n pi:
#  [-4             0 ]
#  [0  -16*cos(4*x2) ]
#  which is negative definite when cos(4*x2)>0 => when n is odd
#  and indefinite when cos(4*x2)>0 => when n is even.
#
# There is thus minima when (x1,x2)=(+/-1, n pi/4) for odd n.
#

x=np.array([2,2])
print('Steepest descent from ',x)
X=np.zeros([2,10])
for i in range(0,10):
    X[:,i]=x
    print(i,'x=',x)
    x=x-0.1*df(x[0],x[1])
plt.plot(X[0,:],X[1,:],'-*',label='SD from (2,2)')
print()

x=np.array([-2,2])
print('Steepest descent from ',x)
X=np.zeros([2,10])
for i in range(0,10):
    X[:,i]=x
    print(i,'x=',x)
    x=x-0.1*df(x[0],x[1])
plt.plot(X[0,:],X[1,:],'-*',label='SD from (-2,2)')
print()

# The other solutions can be found using
# different initial guesses to the steepest descent method.
# Be sure to use a small enough step size, though!

x=np.array([-2,2])
print('Steepest descent from ',x, 'with step size 1')
X=np.zeros([2,10])
for i in range(0,10):
    X[:,i]=x
    print(i,'x=',x)
    x=x-1*df(x[0],x[1])
#plt.plot(X[0,:],X[1,:],'-*')
print('This was a disaster, step size too big...')
print()

plt.pause(0.1)

# Exercise 5.2

# Adding an equality constraint
# h(x1,x2)=x1+x2=0
#
# L = f(x1,x2)+l*h(x1,x2)
#
# Lagrange Multiplier theorem:
# dL/dx1 = df/dx1 + l*dh/dx1 = 0
# dL/dx2 = df/dx2 + l*dh/dx2 = 0
# dL/dl  = h(x1,x2)          = 0
#
# The 3 eqs with 3 unknown:
# dL/dx1 = 4*(x1**2-1)*x1 + l = 0 => 4*(x1**2-1)*x1=-l
# dL/dx2 = -4sin(4*x2) + l = 0    => l=4sin(4*x2)
# dL/dl  = x1+x2  = 0             => x2=-x1
# =>
# x1**3-x1 = sin(4*x1) => x1=-0.846 or x1=0.846
# Thus: (x1,x2)=(-/+ 0.846, +/- 0.846) and l=4sin(4*x2)=(-/+ 0.960)

# Add the line x2=-x1 to the plot
x=np.linspace(-2,2,1000)
y=np.linspace(-2,2,1000)
plt.plot(-x,y,label='h')
# and mark the constrained (candidate) minima
plt.plot(-0.846,0.846,'*',ms=15,label='Optimim point, h=0')
plt.plot(0.846,-0.846,'*',ms=15,label='Optimum point, h=0')
# and plot the cost function along x2=-x1
plt.figure(2)
plt.title('cost function along x2=-x1')
plt.plot(x,f(x,-x),label='f(x_1,-x_1)')
plt.xlabel('x_1')
plt.ylabel('y')
plt.legend()

# Exercise 5.3
# With inequality constraint g(x1,x2)=x1+x2<=0
#
# L = f(x1,x2)+l*g(x1,x2)
# KKT theorem:
# dL/dx1 = df/dx1 + l*dg/dx1 = 0
# dL/dx2 = df/dx2 + l*dg/dx2 = 0
# g(x1,x2)                  <= 0
# l*g(x1,x2)                 = 0
# l                         >= 0
# 

# Case 1: g(x1,x2)=0
# This is the same set of equations as in Exercise 2 and the solutions are
# (x1,x2)=(-/+ 0.846, +/- 0.846) and l=(-/+ 0.960)
# Thus only (x1,x2,l)=(0.846,-0.846,0.960) satisfies all KKT conditions.
# This is a candidate optimum point.
# On the plot from 1a, it is seen that it is indeed a local minimum point.
#
# Case 2: l=0
# This case reduces the to first equations to those in the unconstrained
# case with solution x1=+/- 1 and 4*x2=n pi for odd n.
# Additionally from the KKT conditions g(x1,x2)<=0 which give
# x1+x2 = +/-1+n*pi/4 <= 0 , n odd integer  =>
# n <= -/+ 4/pi  , n odd integer  =>
# n <=  1, n odd for x1=-1
# n <= -3, n odd for x1= 1
#
# The points are now marked on the plot from 1a
fig=plt.figure(1)
plt.plot(-0.846,0.846,'*',ms=6,label='non-KKT point, g=0')
plt.plot(0.846,-0.846,'*',ms=6,label='KKT point, g=0')
for n in range(1,-7,-2):
    x2=n*np.pi/4
    p1=plt.plot(-1,x2,'o',label='KKT point n=%i'%n)
    if n<=-3:
        p2=plt.plot(1,x2,'x',label='KKT point n=%i'%n)
        
plt.legend(fontsize='xx-small',loc='lower center')
fig.canvas.draw_idle() # Needed to update the plot...

plt.show()
