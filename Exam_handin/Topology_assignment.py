import os
from dataclasses import dataclass, field
from typing import Optional, Any
import fenics as fs
import numpy as np
from mma import MMA_petsc #, MMA_numpy #used for debug
from petsc4py import PETSc
#from mpi4py import MPI
import matplotlib.pyplot as plt

from beam_configurator_2d import MaterialProperties2d, GeometryProperties2d, LoadCase2d, CantileverBeam2dLinear, force_to_traction_2d

#setting up the cantilevered beam problem using the data from the assignment
youngs_modulus = 3e9    #[Pa]
poissons_ratio = 0.33   #[-]
length = 1              #[m]
height = 0.1            #[m]
width = 0.1             #[m]
force = -1e3            #[N]
#mesh size is already a standard in the class CantileverBeam2dLinear.

beam_material = MaterialProperties2d(
    e_modulus=youngs_modulus,
    poisson_ratio=poissons_ratio,
    density=None,
)
beam_geometry = GeometryProperties2d(
    length=length,
    height=height,
    thickness=width, #Renamed to thickness in class for clarity
)

beam_loads = LoadCase2d(
    traction=(0, force_to_traction_2d(force, width, height)), #Convert 1kN to traction on surface.
)

beam_solid = CantileverBeam2dLinear(
    material_properties=beam_material,
    geometry_properties=beam_geometry,
    loads=beam_loads,
)

beam_solid.solve()

#Question 1
print("\nQuestion 1:\n")
print(f"Deflection of assign (theoretical): {beam_solid.euler_deflection:.3e}")
print(f"Deflection of assign (measured): {beam_solid.maximum_deflection:.3e}")
print(f"The strain energy: {beam_solid.strain_energy:.3e}")

#Question 2 code answered in this class
#Make new class that inherits CantileverBeamClass and can make holes in the geometry.
@dataclass(kw_only=True)
class HolyBeam(CantileverBeam2dLinear):
    near_zero_val: float = 1e-6 #as specified in hand-in
    _rho_bar: Optional[Any] = field(default=None, init=False, repr=False)
    
    @property
    def rho_bar(self)->fs.Function:
        if self._rho_bar is None:
            hole_function_space = fs.FunctionSpace(self.mesh, "DG", 0)
            self._rho_bar = fs.Function(hole_function_space, name="rho_bar")
            self._rho_bar.vector()[:] = 1.0 #set density fraction to 1 for all elements
        return self._rho_bar

    def set_design(self, radii_vector, near_zero_val=None):
        if near_zero_val is None:
            near_zero_val = self.near_zero_val
        
        radii = np.asarray(radii_vector, dtype=float).ravel() #ensure np.array 1D
        if radii.size !=10:
            raise ValueError(f"The radii vector should be 10 elements long, it was {radii.size} long")
        
        length = self.geometry_properties.length
        height = self.geometry_properties.height
        hole_centers = [(length/10.0*(hole+0.5), height/2.0) for hole in range(radii.size)] #find hole centers using provided formula in question 2.

        Vd = self.rho_bar.function_space()

        class BeamHoles(fs.UserExpression):
            def eval(self, value, x):
                val = 1.0
                for i, (center_x, center_y) in enumerate(hole_centers):
                    if (x[0]-center_x)**2 + (x[1]-center_y)**2 <= radii[i]**2: #formula for circle
                        val = near_zero_val
                        break
                
                value[0] = val

            def value_shape(self):
                return ()

        temp_rho_bar = fs.interpolate(BeamHoles(element=Vd.ufl_element()), Vd).vector()
        self.rho_bar.vector()[:] = temp_rho_bar[:]

        self.reset_state(keep_displacement_field=True)

        return self.rho_bar


    #The next 3 properties handle the adjustability and the method overloads stresses() from CantileverBeam2dLinear         
    @property
    def e_modulus_adjustable(self):
        return fs.Constant(self.material_properties.e_modulus)*self.rho_bar
    
    @property
    def shear_modulus_adjustable(self):
        nu = self.material_properties.poisson_ratio
        return self.e_modulus_adjustable / (2.0*(1.0+nu))
    
    @property
    def lame_lambda_adjustable(self):
        nu = self.material_properties.poisson_ratio
        return self.e_modulus_adjustable*nu/((1+nu)*(1-2*nu))
    
    def stresses(self, trial_function):
        epsilon = self.strains(trial_function)
        dimensions = trial_function.geometric_dimension()

        return self.lame_lambda_adjustable*fs.tr(epsilon)*fs.Identity(dimensions) + 2.0*self.shear_modulus_adjustable*epsilon

    @property
    def material_area_fraction(self): #this property is for question 4 in the hand-in
        length = self.geometry_properties.length
        height = self.geometry_properties.height

        return fs.assemble(self.rho_bar*fs.dx)/(length*height) #formula in q4

    def save_rho_bar(self, *, prefix: str=None, output_dir: str=None, format: str = "pvd"):
        """
        Saves marked areas to boundary_area.pvd by using fenics.File() it returns a bool indicating succesful saving.
        """
        if prefix is None:
            prefix = self.__class__.__name__
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "result_output")
        os.makedirs(output_dir, exist_ok=True) 

        if self._rho_bar is not None:
            fs.File(os.path.join(output_dir, f"{prefix}_rho_bar.{format}")) << self.rho_bar
            return True
        else:
            print("Could not save adjustable density as there is none. Run rho_bar() and try again")
            return False

beam_holy = HolyBeam(
    material_properties=beam_material,
    geometry_properties=beam_geometry,
    loads=beam_loads,
)

hole_radii_vector = np.full(10, height/4.0) #Create full vector of radius 10
beam_holy.set_design(hole_radii_vector) #Set radii
beam_holy.solve()

#question 3 for holy beam
print("\nQuestion 3:\n")
print(f"The maximum displacement value is: {beam_holy.maximum_deflection:.3e}")
print(f"The strain energy is: {beam_holy.strain_energy:.3e}")

#Comparison
print("\nQuestion 4:\n")
print(f"The difference in maximum displacement between \"Full\" and \"Holy\" beam is: {beam_solid.maximum_deflection-beam_holy.maximum_deflection:.3e} m")
print(f"The difference in strain energy between \"Full\" and \"Holy\" beam is: {beam_solid.strain_energy-beam_holy.strain_energy:.3e} J")

#question 4 area fraction of material
print(f"The areal fraction of the used material is: {beam_holy.material_area_fraction:.3e}")
print("\n")

@dataclass(kw_only=True)
class OptimizedBeam(HolyBeam):
    """
    Beam class that optimizes the holes in a beam according to question 5 in the handin.
    It minimizes the strain energy for a beam subject to the area fraction <= 0.8 using MMA.
    The gradients are obtained using first order forward difference approximations.
    """

    area_limit: float = 0.8
    num_radii: int = 10

    min_radius: Optional[Any] = field(default=None, init=False, repr=False)
    max_radius: Optional[Any] = field(default=None, init=False, repr=False)

    _cached_radii: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _cached_objective: Optional[float] = field(default=None, init=False, repr=False)
    _cached_material_fraction: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        height = self.geometry_properties.height
        self.min_radius = np.zeros(self.num_radii, dtype=float)
        self.max_radius = np.full(self.num_radii, 0.5*height-1e-6, dtype=float)

    def clear_cache(self):
        """
        Clears attributes radii, objective and material fraction.
        """
        self._cached_radii = None
        self._cached_objective = None
        self._cached_material_fraction = None

    def constrain_radii(self, radii):
        """
        Constrains the radii to not exceed feasible limits and create invalid geometry/areas
        """
        return np.clip(radii, self.min_radius, self.max_radius)

    def evaluate(self, radii, *, use_cache=True):
        """
        Evaluate the beam with the provided radii by using methods set_design() and solve()
        If cache is set to True then it should retrieve cached values if present to avoid unnecessary computations.
        """
        radii = np.asarray(radii, dtype=float).ravel()
        radii_constrained = self.constrain_radii(radii)

        if (use_cache
            and self._cached_radii is not None
            and np.allclose(radii_constrained, self._cached_radii)):
            #some checks before the attributes are returned
            return self._cached_objective, self._cached_material_fraction

        self.set_design(radii_vector=radii_constrained)
        self.solve()

        self._cached_radii = radii_constrained.copy() #Overwrite caches
        self._cached_objective = float(self.strain_energy)
        self._cached_material_fraction = float(self.material_area_fraction)

        return self._cached_objective, self._cached_material_fraction
    
    def evaluate_f(self, radii, *, use_cache=True):
        """
        Calls evaluate() to get the strain energy and constraint residuals for the optimization problem
        in question 5
        """
        strain_residual, constraint = self.evaluate(radii, use_cache=use_cache)
        constraint_residual = constraint - self.area_limit
        return strain_residual, constraint_residual
    
    def f(self, radii, *, use_cache=True):
        """
        MMA-numpy compatible method that returns evaluated function values
        """
        f0, f1 = self.evaluate_f(radii, use_cache=use_cache)

        return np.array([f0, f1], dtype=float)
    
    def g(self, radii, pertubation: float=2e-3):
        """
        MMA-numpy compatible method that returns gradient around f(radii) using forward finite difference approximation.
        Perturbations of 2mm are used to at least not be under the mesh size for the problem.
        """
        radii = np.asarray(radii, dtype=float).ravel()

        self.clear_cache()
        f_baseline = self.f(radii, use_cache=False) #get f(R)
        jacobian = np.zeros((f_baseline.size, self.num_radii), dtype=float) #for df/dR (R)

        for index, radius in enumerate(radii): #loop over 10 radii and calculate gradients
            radii_perturbed = radii.copy() #to not overwrite radii a deep copy is performed otherwise a view
            radii_perturbed[index] += pertubation #perturb f(R) -> f(R+ΔR) for the ith radius
            f_perturbed = self.f(radii_perturbed, use_cache=False) #evaluate pertubation
            jacobian[:, index] = (f_perturbed - f_baseline)/pertubation #calculate and insert forward FD gradient

        return jacobian

    def f_petsc(self, x: PETSc.Vec):
        """
        Simple wrapper to use PETSc on f()
        Uses Søren's parToLocal in mma.py
        """
        radii_full = MMA_petsc.parToLocal(x) 
        return self.f(radii_full)

    def g_petsc(self, x: PETSc.Vec):
        """
        Simple wrapper to use PETSc on g()
        Uses Søren's parToLocal in mma.py
        """
        radii_full = MMA_petsc.parToLocal(x)
        J_full = self.g(radii_full)

        istart, iend = x.getOwnershipRange()
        return J_full[:, istart:iend]


def run_mma(beam: OptimizedBeam, *, x0_vals=None, kmax=20, move=0.2, xtol=1e-4, ftol=1e-4):
    comm = PETSc.COMM_WORLD

    # Initial guess is standard H/4 unless manually provided
    if x0_vals is None:
        x0_vals = np.full(beam.num_radii, beam.geometry_properties.height/4.0)
    else:
        x0_vals = np.full(beam.num_radii, x0_vals)
    x0 = PETSc.Vec().createMPI(beam.num_radii, comm=comm)

    istart, iend = x0.getOwnershipRange()
    x0.setValues(range(istart, iend), x0_vals[istart:iend])
    x0.assemble()

    mma = MMA_petsc(x0=x0, m=1, f=beam.f_petsc, g=beam.g_petsc) #create instance of MMA_petsc from mma.py

    # Local bounds
    mma.xmin[:] = beam.min_radius[istart:iend]
    mma.xmax[:] = beam.max_radius[istart:iend]

    #Test gradients
    mma.testGradsTaylor()  

    mma.kmax, mma.move, mma.xtol, mma.ftol = kmax, move, xtol, ftol

    xopt = mma.solve(x0)
    return MMA_petsc.parToLocal(xopt)

#Create instance of optimized beam for the optimization problem
beam_optimized = OptimizedBeam(
    material_properties=beam_material,
    geometry_properties=beam_geometry,
    loads=beam_loads,
    verbose=False,
)

#run MMA optimization
xopt_full = run_mma(beam_optimized, kmax=20)

beam_optimized.set_design(xopt_full)
beam_optimized.solve()
beam_optimized.displacement_magnitude

if PETSc.COMM_WORLD.rank == 0: #printouts for question 5
    f0, g1 = beam_optimized.f(xopt_full)
    print("xopt =", xopt_full)
    print("\nQuestion 5:\n")
    print(f"optimized beam displacement = {beam_optimized.maximum_deflection:.3e}")
    print(f"optimized beam strain energy = {f0:3f}")
    print(f"optimized beam constraint (<=0) = {g1:.3e}")
    print(f"optimized beam area fraction = {g1 + beam_optimized.area_limit:.3f}")

    print("\nComparison Full vs Optimized")
    print(f"The difference in maximum displacement between \"Full\" and \"Optimized\" beam is: {beam_solid.maximum_deflection-beam_optimized.maximum_deflection:.3e} m")
    print(f"The difference in strain energy between \"Full\" and \"Optimized\" beam is: {beam_solid.strain_energy-beam_optimized.strain_energy:.3e} J")
    print(f"The difference in area fraction of material between \"Full\" and \"Optimized\" beam is: {1.0-beam_optimized.material_area_fraction:.3e}")

    print("\nComparison Holy vs Optimized")
    print(f"The difference in maximum displacement between \"Holy\" and \"Optimized\" beam is: {beam_holy.maximum_deflection-beam_optimized.maximum_deflection:.3e} m")
    print(f"The difference in strain energy between \"Holy\" and \"Optimized\" beam is: {beam_holy.strain_energy-beam_optimized.strain_energy:.3e} J")
    print(f"The difference in area fraction of material between \"Holy\" and \"Optimized\" beam is: {beam_holy.material_area_fraction-beam_optimized.material_area_fraction:.3e}")

#saving results as pvd
beam_solid.save_result(prefix="solid", output_dir="plots")
beam_solid.save_marked_areas(prefix="solid", output_dir="plots")

beam_holy.save_result(prefix="holy", output_dir="plots")
beam_holy.save_rho_bar(prefix="holy", output_dir="plots")

beam_optimized.save_result(prefix="opt", output_dir="plots")
beam_optimized.save_rho_bar(prefix="opt", output_dir="plots")