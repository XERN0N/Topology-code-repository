# Helper class for solving Ngroup problems in parallel in FEniCS.
# Implemented by Søren Madsen, MPE, Aarhus University.
from fenics import *
from fenics_adjoint import *
import numpy as np
import time

class Parallel():
    # Class for handeling the parallelization of multiple version of the same fenics problem
    # The global_comm is split into Ngroups groups, each suppose to handle one instance of a problem.
    # DOFs needs to be the same, but not necessarily ordered in the same way.
    # Only tested with 'DG' function spaces!
    # mpirun/mpiexec must be used with at least Ngroup CPUs, preferably
    # #CPUs/Ngroup is an integer, e.g., 1.
    
    def __init__(self,global_comm,Ngroups):
        # Sets up the class and communicators for running Ngroups problems on Ngroups groups of CPUs
        self.global_comm = global_comm
        self.Ngroups = Ngroups
        self.global_rank = self.global_comm.Get_rank()
        self.global_size = self.global_comm.Get_size()

        if self.global_size<self.Ngroups:
            if self.global_rank==0:
                print('Not enough MPI processes %i to divide into %i groups.'%(self.global_size,self.Ngroups))
            exit()
        if self.global_size%self.Ngroups != 0 and self.global_rank==0:
            print('Warning: MPI processes %i not divisible by %i groups.'%(self.global_size,self.Ngroups))

        self.group = self.global_rank % self.Ngroups

        # deteremine the group communicators
        self.group_comm = MPI.comm_world.Split(self.group)
        self.group_rank = self.group_comm.Get_rank()
        self.group_size = self.group_comm.Get_size()

    def create_mapping(self,Vglobal,Vgroup):
        # Create mapping from global to group and from group to global DOFs
        t0=time.time()
        self.Vglobal=Vglobal
        self.Vgroup=Vgroup
        self.global_dofs = np.array(self.Vglobal.dofmap().dofs())

        coor_global=np.array(self.Vglobal.tabulate_dof_coordinates())

        coor_all=np.concatenate(self.global_comm.allgather(coor_global))
        dof_all=np.concatenate(self.global_comm.allgather(self.global_dofs))
        
        coor_group=self.Vgroup.tabulate_dof_coordinates()
        dof_group = self.Vgroup.dofmap().dofs()
        # Slow version...
        #self.global2group=[]
        #self.group2global=np.zeros_like(dof_all)
        #i=0
        #for c in coor_group:
        #    idx=np.where( np.sum((coor_all-c)**2,axis=1)<1e-10 )
        #    self.global2group.append(dof_all[idx[0]])
        #    self.group2global[dof_all[idx[0]]]=dof_group[i]
        #    np.delete(coor_all,idx[0],axis=0)
        #    np.delete(dof_all,idx[0],axis=0)
        #    i+=1
        #self.global2group=np.concatenate(self.global2group)
        a=coor_all
        b=coor_group
        if len(b[0])==1:
            point = np.dtype([('x', a.dtype)])
        elif len(b[0])==2:
            point = np.dtype([('x', a.dtype), ('y', a.dtype)])
        elif len(b[0])==3:
            point = np.dtype([('x', a.dtype), ('y', a.dtype), ('z', a.dtype)])
        a_point = a.view(point).squeeze(-1)
        b_point = b.view(point).squeeze(-1)
        m = np.where(np.isin(a_point, b_point, assume_unique=False))[0]
        self.global2group=[]
        self.group2global=np.zeros_like(dof_all)
        for i in range(0,len(m)):
            self.global2group.append(dof_all[m[i]])
            self.group2global[dof_all[m[i]]]=dof_group[i]
            
        self.group2global=sum(self.group_comm.allgather(self.group2global))
        #print(time.time()-t0)
        
    def fglobal2group(self,f_global,f_group):
        # Puts values from function f_global in f_group function
        g = Vector(MPI.comm_world, self.Vglobal.dim())
        g = f_global.vector().gather_on_zero()
        g_all = MPI.comm_world.bcast(g)
        f_group.vector().set_local( g_all[self.global2group] )
        f_group.vector().apply('')

    def vglobal2group(self,v_global,v_group):
        # Puts values from vector v_global in v_group function
        g = Vector(MPI.comm_world, self.Vglobal.dim())
        g = v_global.gather_on_zero()
        g_all = MPI.comm_world.bcast(g)
        v_group.set_local( g_all[self.global2group] )
        v_group.apply('')

    def sgroup2global(self,s):
        # Returns Ngroup scalars from groups to all (global)
        d=np.array(self.global_comm.gather(s))
        if self.global_rank==0:
            g=self.global_comm.bcast(d)
        else:
            g=self.global_comm.bcast(None)
        return g[0:self.Ngroups]
        
    def vgroup2global(self,v_group):
        # Returns values from vector v_group as Ngroup values using global DOFs
        v1=Vector(MPI.comm_self)
        v_group.gather(v1,np.array(range(self.Vgroup.dim()), "intc"))
        v1=v1.get_local()[self.group2global]
        v2=np.array(self.global_comm.gather(v1))
        if self.global_rank==0:
            v3=self.global_comm.bcast(v2)
        else:
            v3=self.global_comm.bcast(None)
        return v3[0:self.Ngroups,self.global_dofs]
