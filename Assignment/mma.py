"""
| MMA implemented by 
| Søren Madsen, Department of Mechanical and Production Engineering, Aarhus University
| 
| Related papers:
|   MMA and GCMMA - two methods for nonlinear optimization, Krister Svanberg, https://people.kth.se/~krille/mmagcmma.pdf
|   Aage, N. & Lazarov, B.S. Struct Multidisc Optim (2013) 47: 493. https://doi.org/10.1007/s00158-012-0869-2
|
| Find test and examples in file mma_test.py
"""
from petsc4py import PETSc
import numpy as np
import scipy as sp
import scipy.sparse
import time
import sys

class MMA_numpy:
    """
|    Method of Moving Asymptotes using numpy
|    min f[0]+a0*z+1/2*b0*z**2+sum_{i=1}^m (c[i]*y[i]+1/2*d[i]*y[i]**2)
|    s.t.
|    f[i]-a[i]*z-y[i] <= 0
|    z>=0, y[i]>=0
|    xmin[j]<=x_j<=xmax[j]
|  
|    Keep d[i]>0 and b0>0 always!
|
|    x0 must be a 1D numpy array.
|    m is the number of constraint functions.
|    Class can be inherited or created. If created, 
|    f, g and plot_k must be passed as keyword arguments to constructor.
|
|    See MMA_petsc for more documentation

    """
    def __init__(self,x0,m,**kwargs):
        self.n=x0.size
        self.m=m
        # f, g and plot_k can be passed as keyword arguments to constructor
        for key,value in kwargs.items():
            if key=='f':
                self.f=value
            elif key=='g':
                self.g=value
            elif key=='plot_k':
                self.plot_k=value
        
        self.output=True

        # Default parameters
        # Change these after calling MMA.__init__
        self.albefa=0.1
        self.move=0.5
        self.asyinit=0.25
        self.asydecr=0.7
        self.asyincr=1.2

        self.epsilon=1e-9
        self.muTol=1e-6
        #self.dualTol=self.muTol/10
        self.llmax=1000
        self.RHOmin=1e-6
        self.xtol=0.01
        self.ftol=0.001
        self.kmax=100
        self.kmin=0
        self.kav=1
        self.lmax=10
        self.xmin=0.0*np.ones(self.n, dtype=float)
        self.xmax=1.0*np.ones(self.n, dtype=float)
        self.a0=1
        self.a=0*np.ones(self.m+1, dtype=float)
        self.b0=1
        self.c=1e3*np.ones(self.m+1, dtype=float)
        self.d=1.0*np.ones(self.m+1, dtype=float) #2*self.c #
        self.z=0
        self.y=0*np.ones(self.m+1, dtype=float)

        self.setNormal()

        # Arrays and variables used
        self.dfp=np.zeros([self.m+1,self.n], dtype=float)
        self.dfm=np.zeros([self.m+1,self.n], dtype=float)
        self.a1=np.zeros(self.n, dtype=float)
        self.a2=np.zeros(self.n, dtype=float)
        self.gamma=np.zeros(self.n, dtype=float)
        self.L=np.zeros(self.n, dtype=float)
        self.U=np.zeros(self.n, dtype=float)
        self.dxL=np.zeros(self.n, dtype=float)
        self.dUx=np.zeros(self.n, dtype=float)
        self.alphaj=np.zeros(self.n, dtype=float)
        self.betaj=np.zeros(self.n, dtype=float)
        self.l=1+np.zeros(self.m+1, dtype=float)
        self.lold=1+np.zeros(self.m+1, dtype=float)
        self.sigma=np.zeros(self.n, dtype=float)
        self.RHO=np.zeros(self.m+1, dtype=float)
        self.x=np.zeros(self.n, dtype=float)
        self.x[:]=x0[:]
        self.xold=np.zeros(self.n, dtype=float)
        self.xk=np.zeros(self.n, dtype=float)
        self.xkold=np.zeros(self.n, dtype=float)
        self.xlold=np.zeros(self.n, dtype=float)
        self.xkolder=np.zeros(self.n, dtype=float)
        self.dfidxj=np.zeros([self.m+1,self.n], dtype=float)
        self.pij=np.zeros([self.m+1,self.n], dtype=float)
        self.qij=np.zeros([self.m+1,self.n], dtype=float)
        self.gij=np.zeros([self.m+1,self.n], dtype=float)
        self.ri=np.zeros([self.m+1], dtype=float)
        self.fi=np.zeros(self.m+1, dtype=float)
        self.fi2=np.zeros(self.m+1, dtype=float)
        self.gi=np.zeros(self.m+1, dtype=float)
        self.wi=np.zeros(self.m+1, dtype=float)
        self.phi=0
        self.dphidl=np.zeros(self.m, dtype=float)
        self.dL=np.zeros(self.n+self.m+1, dtype=float)
        self.dh=np.zeros([self.m,self.n+self.m+1], dtype=float)
        self.mask=np.zeros(self.n+self.m+1)
        self.mma_timing=False
        self.mma_timing2=False
        self.mma_print_inner=False

    def setMinMax(self,np,nq):
        """Set up MMA for min-max (h_i), i=1,...,np with nq 'normal' constraints"""
        self.a0=1
        self.a[0]=0
        self.a[1:np+1]=1
        if nq>0:
            self.a[np+1:np+nq+1]=0
        self.b0=1.0
        self.c[:]=1e3
        self.d[:]=1
        self.z=1.0
        self.y[:]=0

    def setLeastSquare(self):
        """Set up MMA for Least Squares optimization"""
        self.a0=1
        self.a[:]=0
        self.b0=1
        self.c[:]=0
        self.d[:]=1
        self.z=0
        self.y[:]=0
      
    def setNormal(self):
        """Set up MMA for normal optimization of one cost function with constraints"""
        self.a0=1
        self.a[:]=0
        self.b0=1.0
        self.c[:]=1e3
        self.d[:]=1.0
        self.z=0
        self.y[:]=0

    def get_number_of_variables(self):
        return self.n

    def f(self,x):
        """Must be implemented or passed as argument to constructor"""
        return

    def g(self,x):
        """Must be implemented or passed as argument to constructor"""
        return

    def plot_k(self,x):
        """Implement one or passed as argument to constructor to visualize iterations."""
        return

    def plot_l(self,x):
        """Implement one to visualize iterations for dedbugging."""
        return

    def plot_mu(self,x):
        """Implement one to visualize iterations for dedbugging."""
        return

    def plot_ll(self,x):
        """Implement one to visualize iterations for dedbugging."""
        return

    def calc_g(self):
        # calc_phi must be called first (or instead)
        if self.output and self.mma_timing2:
            t0=time.time()

        for i in range(0,self.m+1):
            #self.gij[i,:]=self.pij[i,:]*self.dUx[:]+self.qij[i,:]*self.dxL[:]
            self.gij[i,:]=self.pij[i,:]/(self.U[:]-self.x[:])+self.qij[i,:]/(self.x[:]-self.L[:])
            #self.gi[i]=self.gij[i,:].sum()
            #self.ri[i]=self.fi[i]-((self.pij[i,:]+self.qij[i,:])/self.sigma[:]).sum()
            #self.gi[i]+=self.ri[i]
            self.gi[i]=self.gij[i,:].sum()+self.fi[i]-((self.pij[i,:]+self.qij[i,:])/self.sigma[:]).sum()

        if self.output and self.mma_timing2:
            t1=time.time()
            print('calc_g time: ',t1-t0)

    def calc_phi(self):
        if self.output and self.mma_timing2:
            t0=time.time()
        self.a1[:]=self.pij[0,:]+(np.reshape(self.l[1:],[self.m,1])*self.pij[1:,]).sum(axis=0)
        self.a2[:]=self.qij[0,:]+(np.reshape(self.l[1:],[self.m,1])*self.qij[1:,]).sum(axis=0)
        #da=self.a1[:]-self.a2[:]
        #dLU=self.L[:]-self.U[:]
        #sq=np.maximum(0.0,dLU[:]**2*self.a1[:]*self.a2[:])
        #with np.errstate(divide='ignore',invalid='ignore'):
        #    self.x[:]=np.where( np.logical_and(sq[:]>0,np.absolute(da[:])>self.epsilon),
        #                        (self.L[:]*self.a1[:]-self.U[:]*self.a2[:]+np.sqrt(sq[:]))/da[:],
        #                        0.5*(self.L[:]**2-self.U[:]**2)/dLU[:] )
        self.x[:]=(np.sqrt(self.a1[:])*self.L[:]+np.sqrt(self.a2[:])*self.U[:])/(np.sqrt(self.a1[:])+np.sqrt(self.a2[:]))
        
        if self.output and self.mma_timing2:
            t01=time.time()
            print('calc_phi: calc x time: ',t01-t0)

        self.y[1:]=(self.l[1:]-self.c[1:])/(self.d[1:]+self.epsilon)

        if abs(self.b0)<=self.epsilon:
            self.z=0
        else:
            self.z=(-self.a0+(self.l[:]*self.a[:]).sum())/self.b0

        # For masking out out-of-bounds variables in Hessian
        self.mask[0:self.n]=np.where( np.logical_and(self.x[:]>=self.alphaj[:],self.x[:]<=self.betaj[:]) , 1.0 , 0.0 )
        self.mask[self.n:self.n+self.m]=np.where( self.y[1:]>=0 , 1.0 , 0.0 )
        self.mask[self.n+self.m]=(self.z>=0)
            
        self.z=max(0.0,self.z)
        self.y[1:]=np.maximum(0.0,self.y[1:])
        self.x[:]=np.minimum(np.maximum(self.x[:],self.alphaj[:]),self.betaj[:])
        self.dUx[:]=1.0/(self.U[:]-self.x[:])
        self.dxL[:]=1.0/(self.x[:]-self.L[:])

        self.calc_g()
        self.dphidl[:]=self.gi[1:]-self.a[1:]*self.z-self.y[1:]

        # Not really used for anything, but debugging...
        self.phi=self.gi[0]+(self.l[1:]*self.gi[1:]).sum() \
                  +(self.c[1:]*self.y[1:]+0.5*self.d[1:]*self.y[1:]**2-self.l[1:]*self.y[1:]).sum() \
                  +self.a0*self.z+0.5*self.b0*self.z**2-self.z*(self.a[1:]*self.l[1:]).sum()
        
        if self.output and self.mma_timing2:
            t1=time.time()
            print('calc_phi time: ',t1-t0)


    def solve(self,*args):
        """Solve the optimization problem"""        
        if len(args)>0:
            self.xk[:]=args[0]
        self.dh[:,:]=0.0
        for i in range(1,self.m+1):
            self.dh[i-1,self.n+i-1]=-1 #dh/dyi
            self.dh[i-1,self.n+self.m+1-1]=-self.a[i] #dh/dz
        self.dL[:]=0.0
        self.dL[self.n:self.n+self.m]=self.d[1:self.m+1]
        self.dL[self.n+self.m+1-1]=self.b0

        #d2phix=-np.dot( self.dh, np.dot(np.diag(self.dL),self.dh.transpose()) )
        #print(d2phix)
        #d2phix[:,:]=0
        #for i in range(0,self.m):
        #    #if self.l[i+1]>self.c[i+1]:
        #    #d2phix[i,i]+=-1.0/self.d[i+1]
        #    for j in range(0,self.m):
        #        d2phix[i,j]+=-self.a[i+1]*self.a[j+1]/self.b0
        #print(d2phix)
        #exit()
        
        done=False
        fchange=np.ones(self.kav, dtype=float)
        xchange=np.ones(self.kav, dtype=float)
        k=0
        while ((not done) or k<self.kmin or k<self.kav) and k<self.kmax:
            if k<2:
                #self.sigma[:]=np.minimum(2,0.5*(self.xmax[:]-self.xmin[:]))
                self.sigma[:]=self.asyinit*(self.xmax[:]-self.xmin[:])
            else:
                #for j in range(0,self.n):
                #    gamma=1
                #    if (self.xk[j]-self.xkold[j])*(self.xkold[j]-self.xkolder[j])<0:
                #        gamma=0.7
                #    elif (self.xk[j]-self.xkold[j])*(self.xkold[j]-self.xkolder[j])>0:
                #        gamma=1.2
                #    self.sigma[j]=min( max(self.sigma[j]*gamma,0.01*(self.xmax[j]-self.xmin[j])) , 10*(self.xmax[j]-self.xmin[j]))
                test=(self.xk[:]-self.xkold[:])*(self.xkold[:]-self.xkolder[:])
                self.gamma[:]=np.where( test[:]<0, self.asydecr, np.where( test[:]>0, self.asyincr, 1.0 ) )
                self.sigma[:]=np.minimum( np.maximum(self.sigma[:]*self.gamma[:],0.01*(self.xmax[:]-self.xmin[:])) , 10*(self.xmax[:]-self.xmin[:]) )
            self.L[:]=self.xk[:]-self.sigma[:]
            self.U[:]=self.xk[:]+self.sigma[:]

            self.xkolder[:]=self.xkold[:]
            self.xkold[:]=self.xk[:]
            self.x[:]=self.xk[:]

            if (k==0):
                self.fi=self.f(self.xk)

            if self.output and self.mma_timing:
                t0=time.time()
            self.dfidxj=self.g(self.xk)
            if self.output and self.mma_timing:
                t1=time.time()
                print('calc function grad time: ',t1-t0)
            
            for i in range(0,self.m+1):
                #self.RHO[i]=0
                #for j in range(0,self.n):
                #    self.RHO[i]+=abs(self.dfidxj[i,j])*(self.xmax[j]-self.xmin[j])
                self.RHO[i]=(np.absolute(self.dfidxj[i,:])*(self.xmax[:]-self.xmin[:])).sum()
            self.RHO[:]*=0.1/self.n
            self.RHO[:]=np.maximum(self.RHOmin,self.RHO[:])

            self.alphaj[:]=np.maximum(self.xmin[:],np.maximum(self.L[:]+self.albefa*(self.xk[:]-self.L[:]),self.xk[:]-self.move*(self.xmax[:]-self.xmin[:])))
            self.betaj[:] =np.minimum(self.xmax[:],np.minimum(self.U[:]-self.albefa*(self.U[:]-self.xk[:]),self.xk[:]+self.move*(self.xmax[:]-self.xmin[:])))

            for i in range(0,self.m+1):
                self.dfp[i,:]=np.maximum(0,self.dfidxj[i,:])
                self.dfm[i,:]=np.maximum(0,-self.dfidxj[i,:])

            l=0
            lDone=False
            while not lDone and l<self.lmax:
                self.xlold[:]=self.x[:]
                for i in range(0,self.m+1):
                    self.pij[i,:]=self.sigma[:]**2*(1.001*self.dfp[i,:]+0.001*self.dfm[i,:]+self.RHO[i]/(self.xmax[:]-self.xmin[:]))
                    self.qij[i,:]=self.sigma[:]**2*(0.001*self.dfp[i,:]+1.001*self.dfm[i,:]+self.RHO[i]/(self.xmax[:]-self.xmin[:]))

                # solve dual problem
                if self.output and self.mma_timing:
                    t0=time.time()
                mu=1.0
                self.l[:]=self.c[:]/5.0
                eta=mu/self.l[:]
                ll_total=0
                while mu>self.muTol:
                    done=False
                    ll=0
                    dualTol=0.9*mu
                    self.phi=1e20
                    self.calc_phi()
                    while not done and ll<self.llmax:
                        if self.output and self.mma_timing2:
                            t00=time.time()
                        #self.a1[:]=self.pij[0,:]+(np.reshape(self.l[1:],[self.m,1])*self.pij[1:,]).sum(axis=0)
                        #self.a2[:]=self.qij[0,:]+(np.reshape(self.l[1:],[self.m,1])*self.qij[1:,]).sum(axis=0)
                        self.dL[0:self.n]=2*self.a1[:]*self.dUx[:]**2*self.dUx[:]+2*self.a2[:]*self.dxL[:]**2*self.dxL[:]
                        for i in range(1,self.m+1):
                            self.dh[i-1,0:self.n]=self.pij[i,:]*self.dUx[:]**2-self.qij[i,:]*self.dxL[:]**2
                        if self.output and self.mma_timing2:
                            t11=time.time()
                            print('calc grads time: ',t11-t00)

                            
                        dm=(sp.sparse.diags(self.mask*1.0/self.dL,0,shape=(len(self.dL),len(self.dL)))).tocsr()
                        d2phi=-np.dot( self.dh,dm.dot(self.dh.transpose()) )
                        #d2phi=-np.dot( self.dh, np.dot(np.diag(1.0/self.dL),self.dh.transpose()) )
                        if False:
                            d2phix=-np.dot( self.dh[:,0:self.n], np.dot(np.diag(1.0/self.dL[0:self.n]),self.dh[:,0:self.n].transpose()) )
                            for i in range(0,self.m):
                                if self.l[i+1]>self.c[i+1]:
                                    d2phix[i,i]+=-1.0/self.d[i+1]
                                for j in range(0,self.m):
                                    d2phix[i,j]+=-self.a[i+1]*self.a[j+1]/self.b0
                            #d2phi=d2phix
                        
                        if False:
                            # To check the gradient and hessian
                            def calc_phis(delta):
                                phis=np.zeros([self.m,2])
                                self.calc_phi()
                                phi0=self.phi
                                phis[:,0]=phi0
                                for i in range(1,self.m+1):
                                    self.l[i]+=delta
                                    self.calc_phi()
                                    phis[i-1,1]=self.phi
                                    self.l[i]-=delta
                                self.calc_phi()
                                return phis
                            
                            def calc_hess_diag(delta):
                                phis=np.zeros([self.m,3])
                                self.calc_phi()
                                phis[:,1]=self.phi
                                for i in range(1,self.m+1):
                                    self.l[i]+=delta
                                    self.calc_phi()
                                    phis[i-1,2]=self.phi
                                    
                                    self.l[i]-=2*delta
                                    self.calc_phi()
                                    phis[i-1,0]=self.phi
                                    
                                    self.l[i]+=delta
                                    
                                self.calc_phi()
                                return (phis[:,2]-2*phis[:,1]+phis[:,0])/delta**2
                            
                            def grad(delta):
                                g=np.zeros(self.m)
                                x=calc_phis(delta)
                                g[:]=(x[:,1]-x[:,0])/delta
                                return g

                            def hessian(delta):
                                g0=grad(delta)
                                h=np.zeros([self.m,self.m])
                                for i in range(0,self.m):
                                    self.l[i+1]+=delta
                                    g1=grad(delta)
                                    self.l[i+1]-=delta
                                    h[:,i]=(g1-g0)/delta
                                self.calc_phi()
                                return h

                            delta=1e-4
                            print('xyz', self.x)
                            print('y',self.y[1:])
                            print('z',self.z)
                            print( 'grad' )
                            g=grad(delta)
                            print( g )
                            print( self.dphidl )
                            print( (g-self.dphidl)/self.dphidl )
                            print( 'hess' )
                            h=hessian(delta)
                            print(d2phix)
                            print( calc_hess_diag(delta) )
                            print( h )
                            print( d2phi )
                            print( (h-d2phi)/d2phi )
                            #exit()
                        
                        A=d2phi-np.diag(eta[1:]/self.l[1:])
                        f0=min(-1e-7, 1e-7*np.trace(A)/self.m )
                        #A+=f0*np.identity(self.m)
                        b=-self.dphidl[:]-mu/self.l[1:]

                        delta_l=np.linalg.solve(A,b)
                            
                        delta_eta=-eta[1:]+mu/self.l[1:]-delta_l[:]*eta[1:]/self.l[1:]
                        
                        search=np.dot(self.dphidl+eta[1:],delta_l)+np.dot(self.l[1:],delta_eta)
                        #print(search>0)

                        oneOverStep=max(1.05, max( np.amax( delta_l[:]/(0.01-1)/self.l[1:] ) , np.amax( delta_eta[:]/(0.01-1)/eta[1:] )   ) )
                        step=1.0/oneOverStep
                        self.l[1:]+=step*delta_l[:]
                        eta[1:]+=step*delta_eta[:]
                        
                        self.l=np.maximum(self.epsilon,self.l)
                        eta=np.maximum(self.epsilon,eta)
                        
                        self.calc_phi()
                        dl=np.amax(np.absolute(step*delta_l[:]))
                        de=np.amax(np.absolute(step*delta_eta[:]))
                        res1=np.amax(np.absolute(self.dphidl[:]+eta[1:]))
                        res2=np.amax(np.absolute(self.l[1:]*eta[1:]-mu))
                        done=(res1<dualTol and res2<dualTol) or (dl<self.epsilon and de<self.epsilon)
                        if self.mma_print_inner or (self.output and self.mma_timing2):
                            print("%5d"%ll,"%10f"%mu, "%4f"%step,"phi=%10g"%self.phi,"res1=%12g"%res1,"res2=%12g"%res2,"%12g"%dualTol,"%12g"%np.trace(d2phi),"%12g"%f0,"%12g"%dl,"%12g"%de)

                        if self.output and self.mma_timing2:
                            t21=time.time()
                            print('calc rest time: ',t21-t11)
                        ll+=1
                        ll_total+=1
                        self.plot_ll(self.x)
                    mu*=0.1
                    self.plot_mu(self.x)

                if self.output and self.mma_timing:
                    t1=time.time()
                    print('solve dual time: ',t1-t0,' iter',ll_total)

                self.calc_phi()
                if self.output and self.mma_timing:
                    t0=time.time()

                fi=self.f(self.x)
                if self.output and self.mma_timing:
                    t1=time.time()
                    print('calc f time: ',t1-t0)

                wi=( (self.U[:]-self.L[:])*(self.x[:]-self.xk[:])**2/((self.U[:]-self.x[:])*(self.x[:]-self.L[:])*(self.xmax[:]-self.xmin[:])) ).sum()
                
                if wi>self.epsilon:
                    lDone=True
                    delta=(fi[:]-self.gi[:])/wi
                    for i in range(0,self.m+1):
                        if delta[i]>0:
                            self.RHO[i]=min(10*self.RHO[i],1.1*(self.RHO[i]+delta[i]))
                            lDone=False
                else:
                    lDone=True
                
                #print(self.L,self.U,delta,lDone)
                #self.fi[:]=fi[:]
                self.plot_l(self.x)
                l+=1

            #raw_input("Press Enter to continue...")
            #xchg=np.sqrt( ((self.xk-self.x)**2).sum() )
            xchange[np.mod(k,self.kav)]=abs(self.x-self.xk).max()
            fchange[np.mod(k,self.kav)]=np.amax( np.absolute(fi[:]-self.fi[:]) )
            xchg=np.average(xchange[0:min(self.kav,k+1)])
            fchg=np.average(fchange[0:min(self.kav,k+1)])
            done=( xchg<self.xtol ) and ( fchg<self.ftol )
            self.xk[:]=self.x[:]
            self.fi=fi
            k=k+1
            if self.output:
                print('%4i/%4i'%(k,self.kmax),': lused=%3i'%l,': llused=%5i'%ll_total,'f(x)=%.4f'%fi[0],'f1(x)=%.4f'%fi[1],'max(y)=%.4f'%np.amax(self.y),'z=%.4f'%self.z,'xchg=%.6f'%xchg,'fchg=%.6f'%fchg,' max(x)=%.3f'%np.amax(self.x),'min(x)=%.3f'%np.amin(self.x))
            self.plot_k(self.x)
            sys.stdout.flush()

        return self.x
    
    def testGrads(self,grads=10,components=None):
        """Test the gradients in g against the functions in f using finite differences."""
        print('*** TESTING GRADS ***')
        accu=0.1
        epsilon=0.001
        if components is None:
            # Pick 'grads' random gradient components to test
            P=np.random.rand(grads)*self.n
        else:
            # Test gradient components specified in argument
            P=components
            grads=len(P)
        f0=self.f(self.x)
        dc=self.g(self.x)
        Err=False
        Error=0.0
        Errors=0
        k=0
        for p in P:
            ii=int(p)
            self.x[ii]+=epsilon
            f1=self.f(self.x)
            grad=(f1-f0)/epsilon
            self.x[ii]-=epsilon
            Err=False

            PETSc.Sys.syncPrint('index',ii,'variable value',self.x[ii])
            for j in range(0,self.m+1):
                if abs(dc[j,ii]-grad[j])>max(1e-8,accu*abs(dc[j,ii])):# and abs(dc[j,ii])>self.epsilon:
                    print('***** Errors Found in Gradient! Details Below *****')
                    Err=True
                    Errors+=1
                print('function no.',j,'supplied grad',dc[j,ii],'FD grad',grad[j],'abs. diff.',abs(dc[j,ii]-grad[j]),'allowed diff',max(1e-8,accu*abs(dc[j,ii])))
                Error+=abs(dc[j,ii]-grad[j])
            if not Err:
                print('Grad no.',k+1,' of ',grads,': OK')
            else:
                print('Grad no.',k+1,' of ',grads,': *ERROR*')
            k+=1

        print('Total # of errors: ',Errors,'abs sum of error',Error)
        if Errors==0:
            print('All grad components OK!')
        else:
            print('Some potential errors found in  grad components!')
        return



class MMA_petsc:
    """
|    Method of Moving Asymptotes using petsc
|    min f[0]+a0*z+1/2*b0*z**2+sum_{i=1}^m (c[i]*y[i]+1/2*d[i]*y[i]**2)
|    s.t.
|    f[i]-a[i]*z-y[i] <= 0
|    z>=0, y[i]>=0
|    xmin[j]<=x_j<=xmax[j]
|   
|    x0 must be a PETSc vector.
|    m is the number of constraint functions.
|    Class can be inherited or created. If created, 
|    f, g and plot_k must be passed as keyword arguments to constructor.
|
|    Keep d[i]>0 and b0>0 always!
    """
    def __init__(self,x0,m,**kwargs):
        self.n=x0.getSize()
        self.m=m
        # f, g and plot_k can be passed as keyword arguments to constructor
        for key,value in kwargs.items():
            if key=='f':
                self.f=value
            elif key=='g':
                self.g=value
            elif key=='plot_k':
                self.plot_k=value

        PETSc.Comm(PETSc.COMM_WORLD).Barrier()
        self.rank=PETSc.Comm(PETSc.COMM_WORLD).getRank()
        self.nproc=PETSc.Comm(PETSc.COMM_WORLD).getSize()
        self.output=(self.rank==0)

        # Default parameters
        # Change these after calling MMA.__init__

        #: MMA albefa parameter
        self.albefa=0.1
        #: MMA move parameter
        self.move=0.5
        #: MMA asyinit parameter
        self.asyinit=0.25
        #: MMA asydecr parameter
        self.asydecr=0.7
        #: MMA asyincr parameter
        self.asyincr=1.2

        #: A small number
        self.epsilon=1e-9 #*np.sqrt(self.n+self.m)
        #: Tolerance for solving inner problem
        self.muTol=1e-6
        #self.dualTol=self.muTol/10
        #: Maximum iterations in inner problem        
        self.llmax=1000
        #: MMA Rho0 min parameter 
        self.RHOmin=1e-6
        #: Tolerance for changes in x
        self.xtol=0.01
        #: Tolerance for changes in functions
        self.ftol=0.001
        #: Maximum number of outer iterations
        self.kmax=100
        #: Minimum number of outer iterations
        self.kmin=0
        #: Number of values to average over to check tolerances
        self.kav=1
        #: Maximum number of inner iterations
        self.lmax=10

        #: MMA a0 value
        self.a0=1
        #: MMA a_1 to a_m values (a[0] is not used)
        self.a=0*np.ones(self.m+1, dtype=float)
        #: MMA b0 value
        self.b0=1
        #: MMA c_0 to c_m values (c[0] is not used)
        self.c=1e3*np.ones(self.m+1, dtype=float)         # high c and d guards against infeasible solutions
        #self.d=1.0*np.ones(self.m+1, dtype=float) #
        #: MMA d_0 to d_m values (d[0] is not used)
        self.d=2*self.c
        #: MMA z 
        self.z=0
        #: MMA y_0 to y_m (y[0] is not used)
        self.y=0*np.ones(self.m+1, dtype=float)

        #pmat=PETSc.Mat().createDense(([self.m+1,self.m+1],[PETSc.DECIDE,self.n]))
        #pmat.setUp()
        #pmat.setValues(np.array([0],dtype=np.int32),np.array([5,6,7,8,9,10],dtype=np.int32),np.array([0.5,0.6,0.7,0.8,0.9,1.0]))
        #pmat.setValue(0,10,0.5)
        #pmat.assemblyBegin()
        #pmat.assemblyEnd()
        #for i in range(0,self.m+1):
        #    for j in range(0,self.n):
        #        pmat.setValue(i,j,i+j+0.5)
        #mpi_rank = PETSc.COMM_WORLD.getRank()
        #ilow,ihigh = pmat.getOwnershipRangeColumn()
        #pmat.scale(2.0)
        #PETSc.Sys.syncPrint(pmat.getSize(),pmat.getValue(0,9),pmat.getValue(0,10))
        #PETSc.Sys.syncFlush()
        #exit()

        # Arrays and variables used
        #self.da = PETSc.DA().create([self.m+1, self.n], stencil_width=1)
        #self.x=self.da.createGlobalVec()
        #self.mtmp=self.da.createMatrix()

        self.x=x0.copy()
        self.nlow,self.nhigh = self.x.getOwnershipRange()
        self.nlocal=self.nhigh-self.nlow

        self.xold=self.x.duplicate()
        self.xk=self.x.duplicate()
        self.xkold=self.x.duplicate()
        self.xlold=self.x.duplicate()
        self.xkolder=self.x.duplicate()

        self.vtmp=self.x.duplicate()
        self.vtmp2=self.x.duplicate()
        self.vtmp3=self.x.duplicate()
        self.mtmp=PETSc.Mat().createDense(([self.m+1,self.m+1],[self.nlocal,self.n]))
        self.mtmp.setUp()
        self.mtmp.assemblyBegin()
        self.mtmp.assemblyEnd()
        self.mtmp2=self.mtmp.duplicate()
        self.mtmp3=self.mtmp.duplicate()

        #PETSc.Sys.syncPrint(self.mtmp.getSize(),self.mtmp.getLocalSize())
        #PETSc.Sys.syncFlush()
        #exit()
        
        self.xmin=0.0*np.ones(self.nlocal, dtype=float)
        self.xmax=1.0*np.ones(self.nlocal, dtype=float)
        #self.xmin=PETSc.Vec().createMPI(self.n)
        #self.xmin.set(0.0)
        #self.xmax=PETSc.Vec().createMPI(self.n)
        #self.xmax.set(1.0)

        self.dfp=np.zeros([self.m+1,self.nlocal], dtype=float)
        self.dfm=np.zeros([self.m+1,self.nlocal], dtype=float)
        self.a1=np.zeros(self.nlocal, dtype=float)
        self.a2=np.zeros(self.nlocal, dtype=float)
        self.gamma=np.zeros(self.nlocal, dtype=float)
        self.L=np.zeros(self.nlocal, dtype=float)
        self.U=np.zeros(self.nlocal, dtype=float)
        self.dxL=np.zeros(self.nlocal, dtype=float)
        self.dUx=np.zeros(self.nlocal, dtype=float)
        self.alphaj=np.zeros(self.nlocal, dtype=float)
        self.betaj=np.zeros(self.nlocal, dtype=float)
        self.l=1+np.zeros(self.m+1, dtype=float)
        self.lold=1+np.zeros(self.m+1, dtype=float)
        self.sigma=np.zeros(self.nlocal, dtype=float)
        self.RHO=np.zeros(self.m+1, dtype=float)
        self.dfidxj=np.zeros([self.m+1,self.nlocal], dtype=float)
        self.pij=np.zeros([self.m+1,self.nlocal], dtype=float)
        self.qij=np.zeros([self.m+1,self.nlocal], dtype=float)
        self.gij=np.zeros([self.m+1,self.nlocal], dtype=float)
        self.ri=np.zeros([self.m+1], dtype=float)
        self.fi=np.zeros(self.m+1, dtype=float)
        self.fi2=np.zeros(self.m+1, dtype=float)
        self.gi=np.zeros(self.m+1, dtype=float)
        self.wi=np.zeros(self.m+1, dtype=float)
        self.phi=0
        self.dphidl=np.zeros(self.m, dtype=float)

        #self.dL=np.zeros(self.n+self.m+1, dtype=float)
        #self.dh=np.zeros([self.m,self.n+self.m+1], dtype=float)
        self.dL=np.zeros(self.nlocal, dtype=float)
        #self.dh=np.zeros([self.m,self.nlocal], dtype=float)
        self.dh=np.zeros([self.nlocal,self.m], dtype=float)
        self.dhT=np.zeros([self.nlocal,self.m], dtype=float)
        
        #self.dLPETSc=PETSc.Vec().createMPI([PETSc.DECIDE,self.n+self.m+1])
        #self.dL1=PETSc.Vec().createMPI([self.nlocal,self.n])
        #self.dL2=PETSc.Vec().createSeq([PETSc.DECIDE,self.m+1])
        self.dL1=np.zeros(self.nlocal)
        self.dL2=np.zeros(self.m+1)
        self.maskX=np.zeros(self.nlocal, dtype=float)
        self.maskY=np.zeros(self.m, dtype=float)
        self.maskZ=1.0
        
        self.dh1=PETSc.Vec().create()
        self.dh1.setSizes([self.nlocal,self.n])
        self.dh1.setType('mpi')
        self.dh1.setUp()
        
        #self.dh2=PETSc.Vec().create()
        #self.dh2.setSizes([PETSc.DECIDE,self.m+1])
        #self.dh2.setType('seq')
        #self.dh2.setUp()
        self.dhT1=self.dh1.duplicate()
        #self.dhT2=self.dh2.duplicate()
        
        #self.dhPETSc=PETSc.Mat().create()
        #self.dhPETSc.setSizes(([PETSc.DECIDE,self.m],[PETSc.DECIDE,self.n+self.m+1]))
        ##self.dhPETSc.setType("aij")
        #self.dhPETSc.setType("dense")
        #self.dhPETSc.setUp()
        #self.dhT=PETSc.Mat().create()
        #self.dhT.setSizes(([PETSc.DECIDE,self.n+self.m+1],[PETSc.DECIDE,self.m]))
        ##self.dhT.setType("aij")
        #self.dhT.setType("dense")
        #self.dhT.setUp()
        #self.dLinv=PETSc.Mat().create()
        #self.dLinv.setSizes(([PETSc.DECIDE,self.n+self.m+1],[PETSc.DECIDE,self.n+self.m+1]))
        #self.dLinv.setType("aij")
        #self.dLinv.setUp()

        if False:
            self.dh1=PETSc.Mat().create()
            #self.dh1.setSizes(([self.m,self.m],[self.nlocal,self.n]))
            self.dh1.setSizes(([self.nlocal,self.n],[self.m,self.m])) # Create transpose for right memory partitioning
            self.dh1.setType("dense")
            self.dh1.setUp()
            self.dh1t=PETSc.Mat().create()
            self.dh1t.setSizes(([self.m,self.m],[self.nlocal,self.n]))
            if PETSc.Comm(PETSc.COMM_WORLD).Get_size()==1:
                self.dh1t.setType("dense")
            else:
                self.dh1t.setType("aij")
            self.dh1t.setUp()
            self.dh1T=PETSc.Mat().create()
            self.dh1T.setSizes(([self.nlocal,self.n],[self.m,self.m]))
            self.dh1T.setType("dense")
            self.dh1T.setUp()
            
        self.dh2=PETSc.Mat().create(comm=PETSc.Comm(PETSc.COMM_SELF))
        self.dh2.setSizes(([PETSc.DECIDE,self.m],[PETSc.DECIDE,self.m+1]))
        self.dh2.setType("seqdense")
        self.dh2.setUp()
        self.dh2T=PETSc.Mat().create(comm=PETSc.Comm(PETSc.COMM_SELF))
        self.dh2T.setSizes(([PETSc.DECIDE,self.m+1],[PETSc.DECIDE,self.m]))
        self.dh2T.setType("seqdense")
        self.dh2T.setUp()

        self.d2=PETSc.Mat().create()
        self.d2.setSizes(([PETSc.DECIDE,self.m],[PETSc.DECIDE,self.m]))
        self.d2.setType("dense")
        self.d2.setUp()
        
        self.mma_timing=False
        self.mma_timing2=False
        self.mma_print_inner=False

    def setMinMax(self,np,nq):
        """Set up MMA for min-max (h_i), i=1,...,np with nq 'normal' constraints."""
        self.a0=1
        self.a[0]=0
        self.a[1:np+1]=1
        if nq>0:
            self.a[np+1:np+nq+1]=0
        self.b0=1.0
        self.c[:]=1e3
        self.d[:]=1
        self.z=1.0
        self.y[:]=0
        
    def setLeastSquare(self):
        """Set up MMA for Least Squares optimization."""
        self.a0=1
        self.a[:]=0
        self.b0=1
        self.c[:]=0
        self.d[:]=1
        self.z=0
        self.y[:]=0
      
    def setNormal(self):
        """Set up MMA for normal optimization."""
        self.a0=1
        self.a[:]=0
        self.b0=1
        self.c[:]=1e3
        self.d[:]=1.0
        self.z=0
        self.y[:]=0

    def get_number_of_variables(self):
        return self.n
    
    def f(self,x):
        """Must be implemented or passed as argument to constructor"""
        return

    def g(self,x):
        """Must be implemented or passed as argument to constructor"""
        return

    def plot_k(self,x):
        """Implement one or passed as argument to constructor to visualize or output data in k iterations."""
        return

    def plot_l(self,x):
        """Implement one to visualize or output data in l iterations for dedbugging."""
        return

    def plot_mu(self,x):
        """Implement one to visualize or output data in mu iterations for dedbugging."""
        return

    def plot_ll(self,x):
        """Implement one to visualize or output data in ll iterations for dedbugging."""
        return

    def calc_g(self):
        # calc_phi must be called first (or instead)
        if self.output and self.mma_timing2:
            t0=time.time()

        for i in range(0,self.m+1):
            self.gij[i,:]=self.pij[i,:]/(self.U[:]-self.x.getArray()[:])+self.qij[i,:]/(self.x.getArray()[:]-self.L[:])
            self.vtmp.getArray()[:]=(self.pij[i,:]+self.qij[i,:])/self.sigma[:]
            self.vtmp2.getArray()[:]=self.gij[i,:]
            self.vtmp.assemble()
            self.vtmp2.assemble()
            self.gi[i]=self.vtmp2.sum()+self.fi[i]-self.vtmp.sum()
            
        if self.output and self.mma_timing2:
            t1=time.time()
            print('calc_g time: ',t1-t0)

    def calc_phi(self):
        if self.output and self.mma_timing2:
            t0=time.time()
        self.a1[:]=self.pij[0,:]+(np.reshape(self.l[1:],[self.m,1])*self.pij[1:,]).sum(axis=0)
        self.a2[:]=self.qij[0,:]+(np.reshape(self.l[1:],[self.m,1])*self.qij[1:,]).sum(axis=0)
        #da=self.a1[:]-self.a2[:]
        #dLU=self.L[:]-self.U[:]
        #sq=np.maximum(0.0,dLU[:]**2*self.a1[:]*self.a2[:])
        #if (sq<0).any() or (not np.isfinite(sq).all()):
        #    PETSc.Sys.Print(self.rank,': sq have <0 element(s). Should never happen...')
        ##if (da==0).any() or (not np.isfinite(da).all()):
        ##    PETSc.Sys.Print(self.rank,': da have =0 element(s)') # Does happen!
        #if (dLU==0).any() or (not np.isfinite(dLU).all()):
        #    PETSc.Sys.Print(self.rank,': dLU have =0 element(s)')
        #with np.errstate(divide='ignore',invalid='ignore'):
        #    self.x.getArray()[:]=np.where( np.logical_and(sq[:]>=0,np.absolute(da[:])>self.epsilon),
        #                                   (self.L[:]*self.a1[:]-self.U[:]*self.a2[:]+np.sqrt(sq[:]))/da[:],
        #                                   np.where(np.absolute(dLU[:])>self.epsilon,0.5*(self.L[:]**2-self.U[:]**2)/dLU[:],0.0) )
        ##self.x.getArray()[:]=(self.L[:]*self.a1[:]-self.U[:]*self.a2[:]+np.sqrt(sq[:]))/da[:]
        self.x.getArray()[:]=(np.sqrt(self.a1[:])*self.L[:]+np.sqrt(self.a2[:])*self.U[:])/(np.sqrt(self.a1[:])+np.sqrt(self.a2[:]))
        
        if not np.isfinite(self.x.getArray()).all():
            PETSc.Sys.Print(self.rank,': x have non finite element(s)')

        if self.output and self.mma_timing2:
            t01=time.time()
            print('calc_phi: calc x time: ',t01-t0)

        self.y[:]=(self.l[:]-self.c[:])/(self.d[:]+self.epsilon)

        if abs(self.b0)<=self.epsilon:
            self.z=0
        else:
            self.z=(-self.a0+(self.l[:]*self.a[:]).sum())/self.b0

        self.maskX[:]=np.where( np.logical_and(self.x.getArray()[:]>=self.alphaj[:],self.x.getArray()[:]<=self.betaj[:]) , 1.0 , 0.0 )
        self.maskY[:]=np.where( self.y[1:]>=0 , 1.0 , 0.0 )
        self.maskZ=(self.z>=0)

        self.z=max(0.0,self.z)
        self.y[:]=np.maximum(0.0,self.y[:])
        self.x.getArray()[:]=np.minimum(np.maximum(self.x.getArray()[:],self.alphaj[:]),self.betaj[:])
        self.x.assemble()

        self.dUx[:]=1.0/(self.U[:]-self.x.getArray()[:])
        self.dxL[:]=1.0/(self.x.getArray()[:]-self.L[:])

        self.calc_g()
        self.dphidl[:]=self.gi[1:]-self.a[1:]*self.z-self.y[1:]

        #self.phi=self.gi[0]+(self.l[1:]*self.gi[1:]).sum() 
        # Not really used for anything, but debugging...
        self.phi=self.gi[0]+(self.l[1:]*self.gi[1:]).sum() \
                  +(self.c[1:]*self.y[1:]+0.5*self.d[1:]*self.y[1:]**2-self.l[1:]*self.y[1:]).sum() \
                  +self.a0*self.z+0.5*self.b0*self.z**2-self.z*(self.a[1:]*self.l[1:]).sum()
        
        if self.output and self.mma_timing2:
            t1=time.time()
            print('calc_phi time: ',t1-t0)

    def solve(self,x0):
        """Solve the optimization problem starting from x0"""
        self.xmin[:]=np.minimum(self.xmin,self.xmax-1e-6)
        x0.getArray()[:]=np.minimum(np.maximum(x0.getArray()[:],self.xmin[:]),self.xmax[:])
        x0.assemble()
        x0.copy(self.xk)
        #self.dL[self.n:self.n+self.m]=self.d[1:self.m+1]
        #self.dL[self.n+self.m+1-1]=self.b0
        self.dL2[0:self.m]=self.d[1:self.m+1]
        self.dL2[self.m+1-1]=self.b0
        #self.dh[:,:]=0.0
        #for i in range(1,self.m+1):
        #    self.dh[i-1,self.n+i-1]=1
        #    self.dh[i-1,self.n+self.m+1-1]=-self.a[i]
        for i in range(1,self.m+1):
            self.dh2.setValue(i-1,i-1,-1.0)
            self.dh2.setValue(i-1,self.m,-self.a[i])
            self.dh2T.setValue(i-1,i-1,1.0/self.dL2[i-1])
            self.dh2T.setValue(self.m,i-1,-self.a[i]/self.dL2[self.m])
        self.dh2.assemble()
        self.dh2T.assemble()
        dh2Mat=self.dh2.matMult(self.dh2T)
        dh2=dh2Mat.getDenseArray()
        #self.dL2.setValues(range(1,self.m+1),1.0/self.d[1:self.m+1])
        #self.dL2.setValue(self.m,1.0/self.b0)
        #self.dL2.assemble()
        #PETSc.Sys.syncPrint(self.rank,self.dh2.getDenseArray())
        #PETSc.Sys.syncPrint(self.rank,self.dh2T.getDenseArray())
        #PETSc.Sys.syncPrint(self.rank,self.dL2.getArray())
        #PETSc.Sys.syncFlush()
        done=False
        fchange=np.ones(self.kav, dtype=float)
        xchange=np.ones(self.kav, dtype=float)
        k=0
        while ((not done) or k<self.kmin or k<self.kav) and k<self.kmax:
            if self.output and self.mma_timing:
                tk0=time.time()
            if k<2:
                self.sigma[:]=self.asyinit*(self.xmax[:]-self.xmin[:])
            else:
                test=(self.xk.getArray()[:]-self.xkold.getArray()[:])*(self.xkold.getArray()[:]-self.xkolder.getArray()[:])
                self.gamma[:]=np.where( test[:]<0, self.asydecr, np.where( test[:]>0, self.asyincr, 1.0 ) )
                self.sigma[:]=np.minimum( np.maximum(self.sigma[:]*self.gamma[:],0.01*(self.xmax[:]-self.xmin[:])) , 10*(self.xmax[:]-self.xmin[:]) )
            self.L[:]=self.xk.getArray()[:]-self.sigma[:]
            self.U[:]=self.xk.getArray()[:]+self.sigma[:]

            #self.xkolder[:]=self.xkold[:]
            #self.xkold[:]=self.xk[:]
            #self.x[:]=self.xk[:]
            self.xkold.copy(self.xkolder)
            self.xk.copy(self.xkold)
            self.xk.copy(self.x)

            if (k==0):
                if self.output and self.mma_timing:
                    t0=time.time()
                self.fi=self.f(self.xk)
                if self.output and self.mma_timing:
                    t1=time.time()
                    print('calc f time: ',t1-t0)

            if self.output and self.mma_timing:
                t0=time.time()
            self.dfidxj=self.g(self.xk)
            
            if self.output and self.mma_timing:
                t1=time.time()
                print('calc function grad time: ',t1-t0)

            for i in range(0,self.m+1):
                self.vtmp.getArray()[:]=(np.absolute(self.dfidxj[i,:])*(self.xmax[:]-self.xmin[:]))
                self.vtmp.assemble()
                self.RHO[i]=self.vtmp.sum()
            self.RHO[:]*=0.1/self.n
            self.RHO[:]=np.maximum(self.RHOmin,self.RHO[:])

            self.alphaj[:]=np.maximum(self.xmin[:],np.maximum(self.L[:]+self.albefa*(self.xk.getArray()[:]-self.L[:]),self.xk.getArray()[:]-self.move*(self.xmax[:]-self.xmin[:])))
            self.betaj[:] =np.minimum(self.xmax[:],np.minimum(self.U[:]-self.albefa*(self.U[:]-self.xk.getArray()[:]),self.xk.getArray()[:]+self.move*(self.xmax[:]-self.xmin[:])))

            if False and self.output:
                print('alphaj',self.alphaj)
                print('betaj',self.betaj)

            for i in range(0,self.m+1):
                self.dfp[i,:]=np.maximum(0,self.dfidxj[i,:])
                self.dfm[i,:]=np.maximum(0,-self.dfidxj[i,:])

            l=0
            lDone=False
            while not lDone and l<self.lmax:
                for i in range(0,self.m+1):
                    self.pij[i,:]=self.sigma[:]**2*(1.001*self.dfp[i,:]+0.001*self.dfm[i,:]+self.RHO[i]/(self.xmax[:]-self.xmin[:]))
                    self.qij[i,:]=self.sigma[:]**2*(0.001*self.dfp[i,:]+1.001*self.dfm[i,:]+self.RHO[i]/(self.xmax[:]-self.xmin[:]))

                    if False and self.output:
                        print('pij i=',i,self.pij[i,:])
                        print('qij i=',i,self.qij[i,:])
                        print('dfidxj i=',i,self.dfidxj[i,:])
                # solve dual problem
                if self.output and self.mma_timing:
                    t0=time.time()
                mu=1
                eta=mu/self.l[:]
                ll_total=0
                while mu>self.muTol:
                    done=False
                    ll=0
                    dualTol=0.9*mu
                    #eta=1/self.l[:]
                    self.phi=1e20
                    self.calc_phi()
                    while not done and ll<self.llmax:
                        self.dL1[:]=2*self.a1[:]*self.dUx[:]**2*self.dUx[:]+2*self.a2[:]*self.dxL[:]**2*self.dxL[:]

                        if self.output and self.mma_timing2:
                            t00=time.time()

                        #self.dUx[:]=1.0/(self.U[:]-self.x.getArray()[:])
                        #self.dxL[:]=1.0/(self.x.getArray()[:]-self.L[:])
                        d2phi=np.zeros([self.m,self.m], dtype=float)
                        self.dL2[0:self.m]=self.d[1:self.m+1]
                        self.dL2[self.m+1-1]=self.b0
                        for i in range(1,self.m+1):
                            self.dh2.setValue(i-1,i-1,-1.0)
                            self.dh2.setValue(i-1,self.m,-self.a[i])
                            self.dh2T.setValue(i-1,i-1,-1.0/self.dL2[i-1]*self.maskY[i-1])
                            self.dh2T.setValue(self.m,i-1,-self.a[i]/self.dL2[self.m]*self.maskZ)
                        self.dh2.assemble()
                        self.dh2T.assemble()
                        dh2Mat=self.dh2.matMult(self.dh2T)
                        dh2=dh2Mat.getDenseArray()
                        
                        for i in range(0,self.m):
                            self.dh1.getArray()[:]=self.pij[i+1,:]*self.dUx[:]**2-self.qij[i+1,:]*self.dxL[:]**2
                            self.dh1.assemble()
                            for j in range(0,self.m):
                                #self.dhT1.getArray()[:]=np.where(np.logical_and(self.x.getArray()[:]>self.alphaj[:],
                                #                                                self.x.getArray()[:]<self.betaj[:]),
                                #                                 (self.pij[j+1,:]*self.dUx[:]**2-self.qij[j+1,:]*self.dxL[:]**2)/self.dL1[:],
                                #                                 0.0)
                                self.dhT1.getArray()[:]=self.maskX[:]*(self.pij[j+1,:]*self.dUx[:]**2-self.qij[j+1,:]*self.dxL[:]**2)/self.dL1[:]
                                #self.dhT1.getArray()[:]=(self.pij[j+1,:]*self.dUx[:]**2-self.qij[j+1,:]*self.dxL[:]**2)/self.dL1[:]
                                
                                #self.dhT1.getArray()[:]=(self.pij[j+1,:]*self.dUx[:]**2-self.qij[j+1,:]*self.dxL[:]**2)/self.dL1[:]
                                self.dhT1.assemble()
                                dot=self.dh1.dot(self.dhT1)
                                d2phi[i,j]=-(dot+dh2[i,j])
                                
                        if False:
                            # To check the gradient and hessian
                            def calc_phis(delta):
                                phis=np.zeros([self.m,2])
                                self.calc_phi()
                                phi0=self.phi
                                phis[:,0]=phi0
                                for i in range(1,self.m+1):
                                    self.l[i]+=delta
                                    self.calc_phi()
                                    phis[i-1,1]=self.phi
                                    self.l[i]-=delta
                                self.calc_phi()
                                return phis
                            
                            def calc_hess_diag(delta):
                                phis=np.zeros([self.m,3])
                                self.calc_phi()
                                phis[:,1]=self.phi
                                for i in range(1,self.m+1):
                                    self.l[i]+=delta
                                    self.calc_phi()
                                    phis[i-1,2]=self.phi
                                    
                                    self.l[i]-=2*delta
                                    self.calc_phi()
                                    phis[i-1,0]=self.phi
                                    
                                    self.l[i]+=delta
                                    
                                self.calc_phi()
                                return (phis[:,2]-2*phis[:,1]+phis[:,0])/delta**2
                            
                            def grad(delta):
                                g=np.zeros(self.m)
                                x=calc_phis(delta)
                                g[:]=(x[:,1]-x[:,0])/delta
                                return g

                            def hessian(delta):
                                g0=grad(delta)
                                h=np.zeros([self.m,self.m])
                                for i in range(0,self.m):
                                    self.l[i+1]+=delta
                                    g1=grad(delta)
                                    self.l[i+1]-=delta
                                    h[:,i]=(g1-g0)/delta
                                self.calc_phi()
                                return h

                            delta=1e-4
                            print('xyz', self.x)
                            print('y',self.y[1:])
                            print('z',self.z)
                            print( 'grad' )
                            g=grad(delta)
                            print( g )
                            print( self.dphidl )
                            print( (g-self.dphidl)/self.dphidl )
                            print( 'hess' )
                            h=hessian(delta)
                            print( calc_hess_diag(delta) )
                            print( h )
                            print( d2phi )
                            print( (h-d2phi)/d2phi )
                            #exit()
                        
                        A=d2phi-np.diag(eta[1:]/self.l[1:])
                        f0=min( -1e-7 , 1e-7*np.trace(A)/self.m )
                        #A+=f0*np.identity(self.m)
                        b=-self.dphidl[:]-mu/self.l[1:]
                        
                        delta_l=np.linalg.solve(A,b)

                        delta_eta=(-eta[1:]*self.l[1:]+mu-delta_l[:]*eta[1:])/self.l[1:]

                        search=np.dot(self.dphidl+eta[1:],delta_l)+np.dot(self.l[1:],delta_eta)
                        foundAscent=(search>0)
                        
                        oneOverStep=max(1.05, max( np.amax( delta_l[:]/(0.01-1)/self.l[1:] ) , np.amax( delta_eta[:]/(0.01-1)/eta[1:] )   ) )
                        step=1.0/oneOverStep
                        self.l[1:]+=step*delta_l[:]
                        eta[1:]+=step*delta_eta[:]
                        
                        self.l=np.maximum(self.epsilon/1e3,self.l)
                        eta=np.maximum(self.epsilon/1e3,eta)
                        
                        self.calc_phi()
                        dl=np.amax(np.absolute(step*delta_l[:]))
                        de=np.amax(np.absolute(step*delta_eta[:]))
                        res1=np.amax(np.absolute(self.dphidl[:]+eta[1:]))
                        res2=np.amax(np.absolute(self.l[1:]*eta[1:]-mu))
                        done=(res1<dualTol and res2<dualTol) or (dl<self.epsilon and de<self.epsilon)
                        if (self.mma_print_inner or self.mma_timing2) and self.output:
                            PETSc.Sys.Print("%5d"%ll,"%10f"%mu, "%4f"%step,"phi=%10f"%self.phi,"%12g"%res1,"%12g"%res2,"%12g"%dualTol,"%12g"%np.trace(d2phi),"%12g"%f0,"%12g"%dl,"%12g"%de)
                            PETSc.Sys.syncFlush()

                        if self.output and self.mma_timing2:
                            t21=time.time()
                            print('calc rest time: ',t21-t00)
                        ll+=1
                        ll_total+=1
                        self.plot_ll(self.x)
                    mu*=0.1
                    self.plot_mu(self.x)

                if self.output and self.mma_timing:
                    t1=time.time()
                    print('solve dual time: ',t1-t0,' iter',ll_total)

                self.calc_phi()
                
                if self.output and self.mma_timing:
                    t0=time.time()
                fi=self.f(self.x)
                if self.output and self.mma_timing:
                    t1=time.time()
                    print('calc f time: ',t1-t0)
                    
                self.vtmp.getArray()[:]=(self.U[:]-self.L[:])*(self.x.getArray()[:]-self.xk.getArray()[:])**2/((self.U[:]-self.x.getArray()[:])*(self.x.getArray()[:]-self.L[:])*(self.xmax[:]-self.xmin[:]))
                self.vtmp.assemble()
                wi=self.vtmp.sum()
                
                #dx=abs(self.x-self.xk).max()[1]
                #df=abs(self.fi-fi)
                #if self.rank==0:
                #    print(fi,self.fi,df,dx,wi)
                    
                if wi>self.epsilon:
                    #lDone=True
                    delta=(fi[:]-self.gi[:])/wi
                    #if self.rank==0:
                    #    print('delta',delta)
                    for i in range(0,self.m+1):
                        if delta[i]>0:
                            self.RHO[i]=min(10*self.RHO[i],1.1*(self.RHO[i]+delta[i]))
                            #lDone=False
                    lDone=(fi[:]<=(self.gi[:]+self.epsilon)).all()
                else:
                    lDone=True
                
                self.plot_l(self.x)
                l+=1

            xchange[np.mod(k,self.kav)]=abs(self.x-self.xk).max()[1]
            fchange[np.mod(k,self.kav)]=np.amax( np.absolute(fi[:]-self.fi[:]) )
            xchg=np.average(xchange[0:min(self.kav,k+1)])
            fchg=np.average(fchange[0:min(self.kav,k+1)])
            done=( xchg<self.xtol ) and ( fchg<self.ftol )
            self.x.copy(self.xk)
            self.fi=fi
            k=k+1

            max_x=self.xk.max()[1]
            min_x=self.xk.min()[1]
            if self.output:
                fwi=np.amax(fi[1:])
                index=np.argmax(fi[1:])+1
                PETSc.Sys.Print('%4i/%4i'%(k,self.kmax),': lused=%3i'%l,': llused=%5i'%ll_total,'f0(x)=%.4f'%fi[0],'max(f)=%.4f'%fwi,'i=',index,'max(y)=%.4f'%np.amax(self.y[1:]),'z=%.4f'%self.z,'xchg=%.6f'%xchg,'fchg=%.6f'%fchg,'max(x)=%.3f'%max_x,' min(x)=%.3f'%min_x)

            if self.output and self.mma_timing:
                tplot0=time.time()
            self.plot_k(self.x)
            sys.stdout.flush()
            PETSc.Sys.syncFlush()
            if self.output and self.mma_timing:
                tplot1=time.time()
                print('plot_k time: ',tplot1-tplot0)

            if self.output and self.mma_timing:
                tk1=time.time()
                print('toal iteration time: ',tk1-tk0)
        return self.x

    @staticmethod
    def parToLocal(x):
        # Put whole solution vector in an array on each CPU
        # Useful for testing
        (scatter,vec)=PETSc.Scatter().toAll(x)
        scatter.scatterBegin(x,vec)
        scatter.scatterEnd(x,vec)
        return vec.getArray()

    def testGradsTaylor(self):
        """Test the gradients in g against the functions in f using a Taylor series test."""
        if self.rank==0:
            PETSc.Sys.Print('***                  TAYLOR TEST OF GRADS                   ***')
            PETSc.Sys.Print('*** All factors should be ~4, unless the gradient is linear ***')
        self.x.getArray()[:]+=1e-3
        self.x.assemble()
        f0=self.f(self.x)
        dg=self.g(self.x)
        dg_petsc=self.x.duplicate()
        dx=self.x.duplicate()
        dx.getArray()[:]=1
        dx.assemble()
        x1=self.x.duplicate()
        h=0.01
        nErrors=5
        Errors=np.zeros([nErrors,self.m+1])
        for i in range(0,nErrors):
            x1.getArray()[:]=self.x.getArray()[:]+h*dx.getArray()[:]
            x1.assemble()
            f1=self.f(x1)
            for j in range(0,self.m+1):
                dg_petsc.getArray()[:]=dg[j,:]
                dg_petsc.assemble()
                d=dg_petsc.dot(dx)
                Errors[i,j]=np.abs(f1[j]-f0[j]-h*d)
            if self.rank==0:
                with np.printoptions(formatter={'float': '{: 0.3f}'.format}):
                    PETSc.Sys.syncPrint('i=%3i'%i,'h=%8f'%h,'Errors=',Errors[i,:])
                    if i>=1:
                        #PETSc.Sys.syncPrint('i=%3i'%i,'h=%8f'%h,'Rate  =',np.log(Errors[i-1,:]/Errors[i,:])/np.log(2))
                        PETSc.Sys.syncPrint('i=%3i'%i,'h=%8f'%h, 'Factor=',np.where(Errors[i-1]>1e-15,Errors[i-1,:]/Errors[i,:],0))
                PETSc.Sys.syncPrint()
            PETSc.Sys.syncFlush()
            h=h/2
    
    def testGrads(self,grads=10,components=None):
        """Test the gradients in g against the functions in f using finite differences."""
        if self.rank==0:
            PETSc.Sys.Print('*** TESTING GRADS ***')
        accu=0.1
        epsilon=0.001
        Pv=PETSc.Vec().createMPI(grads)
        if components is None:
            # Pick 'grads' random gradient components to test
            Pv.getArray()[:]=np.random.rand(Pv.getArray()[:].size)*self.n
            Pv.assemble()
            P=self.parToLocal(Pv)
        else:
            # Test gradient components specified in argument
            P=components
            grads=len(P)
        #PETSc.Sys.syncPrint(self.rank,P)
        #PETSc.Sys.syncFlush()
        Errors=PETSc.Vec().create()
        Errors.setSizes([1,self.nproc])
        Errors.setType('mpi')
        Errors.setUp()
        #iErr,jErr=Errors.getOwnershipRange()
        Error=PETSc.Vec().create()
        Error.setSizes([1,self.nproc])
        Error.setType('mpi')
        Error.setUp()
        #print self.rank,self.x.getArray()[1]
        f0=self.f(self.x)
        dc=self.g(self.x)
        Err=False
        k=0
        Errors.getArray()[0]=0
        Error.getArray()[0]=0
        for p in P:
            i=int(p)
            ii=i-self.nlow
            if i>=self.nlow and i<self.nhigh:
                self.x.getArray()[ii]+=epsilon
            self.x.assemble()
            f1=self.f(self.x)
            grad=(f1-f0)/epsilon
            if i>=self.nlow and i<self.nhigh:
                self.x.getArray()[ii]-=epsilon
            self.x.assemble()
            Err=False
            if i>=self.nlow and i<self.nhigh:
                PETSc.Sys.syncPrint('rank=',self.rank,'nlow=',self.nlow,'nhigh=',self.nhigh,'index',i,'variable value',self.x.getArray()[ii])
                for j in range(0,self.m+1):
                    if abs(dc[j,ii]-grad[j])>max(1e-8,accu*abs(dc[j,ii])):# and abs(dc[j,ii])>self.epsilon:
                        PETSc.Sys.syncPrint('***** Errors Found in Gradient! Details Below *****')
                        Err=True
                        Errors.getArray()[0]+=1
                    PETSc.Sys.syncPrint('function no.',j,'supplied grad',dc[j,ii],'FD grad',grad[j],'abs. diff.',abs(dc[j,ii]-grad[j]),'allowed diff',max(1e-8,accu*abs(dc[j,ii])))
                    Error.getArray()[0]+=abs(dc[j,ii]-grad[j])
                if not Err:
                    PETSc.Sys.syncPrint('Grad no.',k+1,' of ',grads,': OK')
                else:
                    PETSc.Sys.syncPrint('Grad no.',k+1,' of ',grads,': *ERROR*')
                PETSc.Sys.syncPrint()
            PETSc.Sys.syncFlush()
            k+=1

        self.x.assemble()
        Errors.assemble()
        nErr=Errors.sum()
        Error.assemble()
        Err=Error.sum()

        if self.rank==0:
            PETSc.Sys.Print('Total # of errors: ',nErr,'abs sum of error',Err)
            if nErr==0:
                PETSc.Sys.Print('All grad components OK!')
            else:
                PETSc.Sys.Print('Some potential errors found in  grad components!')
            PETSc.Sys.Print('*** DONE TESTING GRADS ***')
        PETSc.Sys.syncFlush()
        return

