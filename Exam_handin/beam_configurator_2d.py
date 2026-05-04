import os
from dataclasses import dataclass, field, fields
from typing import Optional, Tuple, Dict, Any
import fenics as fs

@dataclass(frozen=True, slots=True, kw_only=True)
class MaterialProperties2d:
    """
        Dataclass containing the material properties for a 2D fenics problem.

    Attributes:
        density:        Material density p  [kg/m³]
        e_modulus:      Young's modulus     [Pa]
        poisson_ratio:  v                   [-]
    Properties:
        shear_modulus:  µ                   [Pa] 
    Notes:
        µ calculated from poisson's ratio and E-modulus.
    Raises:
        ValueError: When inputs are negative or poisson ratio is outside (0,0.5].
    """
    density: float
    e_modulus: float
    poisson_ratio: float
    
    _shear_modulus: Optional[float] = field(default=None, init=False)

    @property
    def shear_modulus(self):
        if self._shear_modulus is not None:
            return self._shear_modulus
        else:
            shear_modulus_calculated = self.e_modulus / (2.0*(1.0+self.poisson_ratio))
            object.__setattr__(self, "_shear_modulus", shear_modulus_calculated)
        return self._shear_modulus

    @property
    def lame_lambda(self):
        return self.e_modulus*self.poisson_ratio/((1+self.poisson_ratio)*(1-2*self.poisson_ratio))

    def __post_init__(self):
        if self.e_modulus <= 0:
            raise ValueError("Youngs modulus cannot be <= 0")
        if self.poisson_ratio <= 0 or self.poisson_ratio > 0.5:
            raise ValueError(f"Poisson ratio cannot be {self.poisson_ratio} as it should be within 0 and 0.5")
        if  self.shear_modulus <= 0:
            raise ValueError(f"Shear modulus must be above 0. Shear modulus was {self.shear_modulus}.")

@dataclass(frozen=True, slots=True, kw_only=True)
class GeometryProperties2d:
    """
    Dataclass containing the geometry properties for a 2D fenics problem.

    Attributes:
        length:         [m]
        height:         [m]
        thickness:      [m]
    Properties:
        volume:         [m³]
        area_moment:    [m⁴]
        section_area:   [m²]
    Raises:
        ValueError: When inputs are negative.
    """
    length: float
    height: float
    thickness: float
    
    @property
    def volume(self)->float:
        return self.length*self.height*self.thickness
    
    @property
    def area_moment(self):
        return self.thickness*self.height**3/12
    
    @property
    def section_area(self):
        return self.height*self.thickness
    
    def __post_init__(self): #This function checks for values being > 0.
        for attribute in fields(self):
            if getattr(self, attribute.name) <= 0:
                raise ValueError(f"The attribute {str(attribute.name)} has negative values of {getattr(self, attribute.name)}")

@dataclass(frozen=True, slots=True, kw_only=True)
class LoadCase2d:
    """
    Dataclass storing the load cases (traction and body forces) as fenics.Constant to be used in a beam class. 
    The coordinate directions x and y are positive right and up respectively.

    Attributes:
        gravity_accel:
        gravity_dir:
        general_force:
        traction:
        name:
    Properties:
        traction_forces:
    Notes:
        Use traction_forces to get the fenics.Constant object.
    Raises:
        RuntimeError: When use_gravity is False and body_forces() is called.
        ValueError: When traction is not provided and traction_forces() is called.
    """
    use_gravity: bool = False
    gravity_accel: float = 9.81
    gravity_dir: Optional[Tuple[float, float]] = None
    general_force: Optional[Dict] = None
    traction: Optional[Tuple[float, float]] = None
    name: Optional[str] = None

    def __post_init__(self): #This function handles not implemented general force and informs about gravity
        if self.general_force is not None:
            raise NotImplementedError("This has not yet been implemented")
        if self.gravity_dir not in ((0.0, -1.0), None) or self.gravity_accel != 9.81:
            print(f"The gravity direction is {self.gravity_dir} with magnitude {self.gravity_accel}")

    def body_forces(self, density: float=None):
        """
        Calculates body forces and returns fs.Constant object.

        Raises:
            RuntimeError: When use_gravity is False and body_forces() is called.
        """
        if self.use_gravity:
            gravity_force = self.gravity_accel*density
            if self.gravity_dir is None:
                return fs.Constant((0.0, -gravity_force))    
            else:
                gx, gy = self.gravity_dir
                return fs.Constant((gx*gravity_force, gy*gravity_force)) 
        else:
            raise RuntimeError(f"use_gravity is {self.use_gravity} but body_forces was called. Decide if gravity is present and try again")
    
    @property
    def traction_forces(self):
        """
        Generates fenics.Constant object from the attribute traction.
        
        Raises:
            ValueError: When traction is not provided.
        """
        if self.traction is not None:
            return fs.Constant(self.traction)
        else:
            raise ValueError("Traction not specified")
        
@dataclass(kw_only=True)
class CantileverBeam2dLinear:
    """
    Dataclass that stores and calculates the standard cantilevered beam with left side fixed and right side load.
    fixed from (0.0,0.0) to (0.0, height) and load from (length,0.0) to (length, height).
    use material and geometry classes to describe properties.
    This class works for  

    Attributes:
        material_properties:    class like material_properties_2d
        geometry_properties:    class like geometry_properties_2d
        loads:
        mesh_size:
    Properties:
        mesh:                   fenics.RectangleMesh
        von_mises:              fenics.Projection onto post-processing scalar field
        displacement_magnitude: fenics.Projection onto post-processing scalar field
        euler_deflection:       float   - calculate y-deflection from standard formula dir- is down
        is_linear:              bool    - is deflection within 1% of length
    Notes:
        y-direction is positive upwards meaning that deflection downwards will be negative.
    Raises:
        RuntimeError: When von_mises or displacement_magnitude is run before solve()
    """
    material_properties: MaterialProperties2d
    geometry_properties: GeometryProperties2d
    loads: LoadCase2d
    mesh_size: Optional[Tuple[int, int]] = None
    verbose: bool = True

    _already_built: bool = field(default=False, init=False, repr=False)
    _mesh: Optional[Any] = field(default=None, init=False, repr=False)
    _function_space: Optional[Any] = field(default=None, init=False, repr=False)
    _trial_function: Optional[Any] = field(default=None, init=False, repr=False)
    _test_function: Optional[Any] = field(default=None, init=False, repr=False)

    _F_constant: Optional[Any] = field(default=None, init=False, repr=False)
    _T_constant: Optional[Any] = field(default=None, init=False, repr=False)
    _boundary_area: Optional[Any] = field(default=None, init=False, repr=False)
    _boundary_conditions: Optional[Any] = field(default=None, init=False, repr=False)
    _boundary_applied_areas: Optional[Any] = field(default=None, init=False, repr=False)

    _displacement_field: Optional[Any] = field(default=None, init=False, repr=False)
    _bilinear_lhs: Optional[Any] = field(default=None, repr=False)
    _loading_rhs: Optional[Any] = field(default=None, repr=False)

    _post_scalar_field: Optional[Any] = field(default=None, init=False, repr=False)
    _strain_energy: Optional[Any] = field(default=None, init=False, repr=False)
    _von_mises: Optional[Any] = field(default=None, init=False, repr=False)
    _displacement_magnitude: Optional[Any] = field(default=None, init=False, repr=False)

    def __post_init__(self): #Initializes mesh size to height/50 if it has not been set already.
        if self.mesh_size is None:
            size = self.geometry_properties.height/50
            mesh_number_length = int(self.geometry_properties.length/size)
            mesh_number_height = int(self.geometry_properties.height/size)
            self.mesh_size = (mesh_number_length, mesh_number_height)
            
            if self.verbose:
                print(f"Mesh size autocalculated to {size*1000} mm for x and y")
        else:
            if self.verbose:
                print(f"Mesh size manually set to {self.mesh_size} [m]")

    @staticmethod
    def strains(trial_function):
        """
        Calculates strains for the trial function u.
        """
        return 0.5*(fs.grad(trial_function)+fs.grad(trial_function).T)    

    def stresses(self, trial_function):
        """
        Calculates stresses for the trial function u.
        """
        epsilon = self.strains(trial_function)
        shear_modulus = self.material_properties.shear_modulus
        dimensions = trial_function.geometric_dimension()
        lame_lambda = self.material_properties.lame_lambda
        return lame_lambda*fs.tr(epsilon)*fs.Identity(dimensions)+2*shear_modulus*epsilon

    @property
    def mesh(self):
        if self._mesh is not None:
            return self._mesh
        else:
            length = self.geometry_properties.length
            height = self.geometry_properties.height
            self._mesh = fs.RectangleMesh(fs.Point(0,0), fs.Point(length, height), *self.mesh_size, diagonal="right")
        return self._mesh

    def _create_function_spaces(self, type="P", degree=1):
        """
        create function spaces for both the vector valued space used in the problem and scalar valued space for post processing.
        """
        if self._function_space is None:
            self._function_space = fs.VectorFunctionSpace(self.mesh, type, degree)
        if self._post_scalar_field is None:
            self._post_scalar_field = fs.FunctionSpace(self.mesh, type, degree)

    def _get_boundary_area(self):
        """
        Creates boundary area ds (fenics.MeshFunction) if not present and returns ds. It auto sets everything to 0.
        """
        if self._boundary_area is None:
            mesh_dimensionality = self.mesh.topology().dim() - 1
            self._boundary_area = fs.MeshFunction("size_t", self.mesh, mesh_dimensionality, 0)
        return self._boundary_area

    def _mark_end_faces(self, tol=1e-10):
        """
        Marks end faces from get_boundary_area() with fixed face (1) and load face (2).
        returns the marked end faces which can be further specified with 1 or 2.
        """
        marking_areas = self._get_boundary_area()
        marking_areas.set_all(0)
        length = self.geometry_properties.length

        class fixed_boundary(fs.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fs.near(x[0], 0.0, tol)
            
        class load_boundary(fs.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fs.near(x[0], length, tol)
            
        fixed_boundary().mark(marking_areas, 1)
        load_boundary().mark(marking_areas, 2)

        self._boundary_applied_areas = fs.Measure("ds", domain=self.mesh, subdomain_data=marking_areas)

        return marking_areas

    def _build_problem(self):
        """
        Creates function spaces, marks end faces, applies boundary conditions and loads and sets up lhs and rhs of equation to solve.
        It rewrites from cantilever_beam_2d_linear attribute naming to Søren's/fenics notation for the equations.
        """
        if self._already_built:
            return

        self._create_function_spaces()
        marked_areas = self._mark_end_faces()

        self._trial_function = fs.TrialFunction(self._function_space)
        self._test_function = fs.TestFunction(self._function_space)
        self._boundary_conditions = fs.DirichletBC(self._function_space, fs.Constant((0.0,0.0)), marked_areas, 1)
        
        if self.loads.use_gravity:
            self._F_constant = self.loads.body_forces(self.material_properties.density)
        else:
            self._F_constant = fs.Constant((0.0,0.0))
        self._T_constant = self.loads.traction_forces

        #Local variables that corresponds to Søren's/fenics notation style
        u = self._trial_function
        v = self._test_function
        #bc = self._boundary_conditions #not used
        F = self._F_constant
        T = self._T_constant
        ds = self._boundary_applied_areas
        sigma = self.stresses
        epsilon = self.strains
        t = self.geometry_properties.thickness

        self._bilinear_lhs = t*fs.inner(sigma(u), epsilon(v))*fs.dx   #a in a==L
        self._loading_rhs = t*fs.dot(F, v)*fs.dx + t*fs.dot(T, v)*ds(2) #L in a==L

        self._already_built = True

    def solve(self, solver_settings: Optional[Dict] = None, name="Displacement"):
        """
        Solve system by building the problem and using fenics.solve(a==L, u, bc, solver_properties).
        solver_settings passes dict to solver_properties in fenics.solve()
        returns displacement field u.
        """
        if solver_settings is None:
            solver_settings = {"linear_solver": "cg", "preconditioner": "hypre_amg"}
        if not self._already_built:
            self._build_problem()
        if self._displacement_field is None:
            self._displacement_field = fs.Function(self._function_space, name=name)
        else:
            self._displacement_field.rename(name, name)
        if self.verbose:
            print('DOFs: %i'%self._function_space.dim())

        fs.solve(self._bilinear_lhs == self._loading_rhs,
                 self._displacement_field,
                 self._boundary_conditions,
                 solver_parameters=solver_settings
                 )
        
        self.reset_state(keep_displacement_field=True) #delete old computed values.

        return self._displacement_field
    
    def reset_state(self, keep_displacement_field=True):
        if not keep_displacement_field:
            self._displacement_field = None

        self._strain_energy = None
        self._von_mises = None
        self._displacement_magnitude = None

    def reset_problem(self, keep_mesh=True):
        """
        Resets all relevant variables for the problem.
        reset_state is called in this method.
        """
        if not keep_mesh:
            self._mesh = None
        self.reset_state(keep_displacement_field=False)

        self._already_built=False

        self._function_space=None
        self._post_scalar_field=None
        self._trial_function=None
        self._test_function=None

        self._boundary_area = None
        self._boundary_applied_areas = None
        self._boundary_conditions = None

        self._F_constant = None
        self._T_constant = None

        self._bilinear_lhs = None
        self._loading_rhs = None

        if self.verbose:
            print("The problem has been reset and the instance is ready for new problem")
        
    @property
    def strain_energy(self):
        if self._displacement_field is not None:
            if self._strain_energy is None:
                t = self.geometry_properties.thickness
                u = self._displacement_field
                pot_energy = 0.5*t*fs.inner(self.stresses(u), self.strains(u))*fs.dx
                self._strain_energy = float(fs.assemble(pot_energy))
            return self._strain_energy
        else:
            raise RuntimeError("run solve() before calculating strain energy")

    @property #TODO fix
    def von_mises(self):
        if self._von_mises is not None:
            return self._von_mises
        elif self._displacement_field is None or self._post_scalar_field is None:
            raise RuntimeError("Please run solve() first before calculating displacement magnitudes")
        else:   
            sigma = self.stresses(self._displacement_field)
            dimensions = self._displacement_field.geometric_dimension()
            deviatoric_stress = sigma - (1.0/3.0)*fs.tr(sigma)*fs.Identity(dimensions)
            von_mises_stress = fs.sqrt(3.0/2*fs.inner(deviatoric_stress, deviatoric_stress))
            self._von_mises = fs.project(von_mises_stress, self._post_scalar_field)
            self._von_mises.rename("von Mises", "von Mises stress")
            return self._von_mises
    
    @property
    def displacement_magnitude(self):
        if self._displacement_magnitude is not None:
            return self._displacement_magnitude
        elif self._displacement_field is None or self._post_scalar_field is None:
            raise RuntimeError("Please run solve() first before calculating displacement magnitudes")
        else:
            displacement_magnitude = fs.sqrt(fs.dot(self._displacement_field, self._displacement_field))
            self._displacement_magnitude = fs.project(displacement_magnitude, self._post_scalar_field)
            self._displacement_magnitude.rename("magnitude", "displacement magnitude")
            return self._displacement_magnitude
        
    @property
    def maximum_deflection(self):
        if self._displacement_magnitude is not None:
            return self._displacement_magnitude.vector().max()
        else:
            return self.displacement_magnitude.vector().max()
    
    @property
    def euler_deflection(self):
        force = self.loads.traction[1] * self.geometry_properties.section_area
        length, area_moment = self.geometry_properties.length, self.geometry_properties.area_moment
        e_modulus = self.material_properties.e_modulus
        deflection = force*length**3/(3*e_modulus*area_moment)
        if abs(deflection) > 0.01*length and self.verbose:
            print(f"WARNING: large deformations detected! Vertical deformation is {100*abs(deflection)/length:.3f}%")
        return deflection
    
    @property
    def is_linear(self):
        deflection = self.euler_deflection
        if abs(deflection) > abs(0.01*self.geometry_properties.length):
            return False
        else:
            return True
    
    def save_result(self, *, prefix: str=None, output_dir: str=None, format: str = "pvd"):
        """
        Saves result to displacement, displacement_magnitude and von_mises.pvd using fenics.File().
        returns tuple of bools (disp_save, mag_save, von_mises_save) with indicates if the saving was succesful.
        """
        displacement_save: bool = False
        magnitude_save: bool = False
        von_mises_save: bool = False
        
        if prefix is None:
            prefix = self.__class__.__name__
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "result_output")
        os.makedirs(output_dir, exist_ok=True) 
            
        if self._displacement_field is not None:
            displacement_save: bool = True
            fs.File(os.path.join(output_dir, f"{prefix}_displacement.{format}")) << self._displacement_field 
        if self._displacement_magnitude is not None:
            magnitude_save: bool = True
            fs.File(os.path.join(os.path.join(output_dir, f"{prefix}_displacement_magnitude.{format}"))) << self._displacement_magnitude
        if self._von_mises is not None:
            von_mises_save: bool = True
            fs.File(os.path.join(os.path.join(output_dir, f"{prefix}_von_mises.{format}"))) << self._von_mises
        return (displacement_save, magnitude_save, von_mises_save)
    
    def save_marked_areas(self, *, prefix: str=None, output_dir: str=None, format: str = "pvd"):
        """
        Saves marked areas to boundary_area.pvd by using fenics.File() it returns a bool indicating succesful saving.
        """
        if prefix is None:
            prefix = self.__class__.__name__
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "result_output")
        os.makedirs(output_dir, exist_ok=True) 

        if self._boundary_area is not None:
            fs.File(os.path.join(output_dir, f"{prefix}_masked_areas.{format}")) << self._boundary_area
            return True
        else:
            print("Could not save boundary area as there is none. Mark area (solve) and try again")
            return False
        
def force_to_traction_2d(force:float, width:float, height:float)-> float:
    """
    Calculates the traction needed for a given section profile and force.
    All inputs are in SI [N] | [m]
    """
    area = width*height
    return force/area

if __name__ == "__main__":
    aluminum = MaterialProperties2d(e_modulus=71e9, poisson_ratio=0.33, density=2700.0)
    PVC = MaterialProperties2d(e_modulus=3e9, poisson_ratio=0.33, density=1380.0)
    beam_rectangle_large = GeometryProperties2d(height=0.1, thickness=0.1, length=1.0)
    load = LoadCase2d(traction=(0.0, -1.0e5))

    beam_PVC = CantileverBeam2dLinear(material_properties=PVC,
                                               geometry_properties=beam_rectangle_large,
                                               loads=load)
    
    beam_PVC.solve()
    beam_PVC.save_result()
    beam_PVC.save_marked_areas()
    print(beam_PVC.maximum_deflection)
    print(force_to_traction_2d(-1000, 0.1, 0.1))

    beam_alu = CantileverBeam2dLinear(material_properties=aluminum,
                                               geometry_properties=beam_rectangle_large,
                                               loads=load)
    
    beam_alu.solve()
    print(beam_alu.maximum_deflection)