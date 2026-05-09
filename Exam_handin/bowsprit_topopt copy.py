"""
Topology optimization of a simplified bowsprit beam for the 2026 exam.

The model is a 2D linear-elastic cantilever with two angled top-surface
tractions. The optimizer uses SIMP, double filtering, pyadjoint, and MMA.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import fenics as fs
import fenics_adjoint as fa
import numpy as np
from petsc4py import PETSc
from ufl import tanh

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from beam_configurator_2d import (
    CantileverBeam2dLinear,
    GeometryProperties2d,
    LoadCase2d,
    MaterialProperties2d,
)
from mma import MMA_petsc


CLAMP_MARKER = 1
F1_MARKER = 2
F2_MARKER = 3


@dataclass(frozen=True, slots=True, kw_only=True)
class BowspritLoadProperties:
    """
    Dataclass storing the simplified bowsprit loads.

    F1 is applied on the top surface near x = 0.2L.
    F2 is applied on the top surface from x = 0.95L to x = L.
    """
    f1_total_force: float = 2000.0
    f2_total_force: float = 3000.0
    f1_angle_deg: float = 65.0
    f2_angle_deg: float = 55.0
    f1_x_center_frac: float = 0.20
    f1_strip_half_frac: float = 0.05
    f2_x_min_frac: float = 0.95
    f2_x_max_frac: float = 1.00

    def __post_init__(self):
        if self.f1_total_force <= 0 or self.f2_total_force <= 0:
            raise ValueError("Bowsprit forces must be positive.")
        if self.f1_strip_half_frac <= 0:
            raise ValueError("The F1 strip half width must be positive.")
        if not 0.0 < self.f1_x_center_frac < 1.0:
            raise ValueError("The F1 load center must be inside the beam length.")
        if not 0.0 <= self.f2_x_min_frac < self.f2_x_max_frac <= 1.0:
            raise ValueError("The F2 load strip must satisfy 0 <= x_min < x_max <= 1.")


@dataclass(frozen=True, slots=True)
class OptimizationSettings:
    """
    Tunable parameters for the exam optimization run.
    """
    mesh_size: Tuple[int, int] = (230, 30)
    filter_radius: float = 0.03
    volume_fraction: float = 0.25
    pitch_weight_alpha: float = 2.5
    beta_schedule: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    kmax: int = 40
    move: float = 0.1
    eta_d: float = 0.45
    eta_i: float = 0.5
    eta_e: float = 0.55

    @property
    def eta3(self):
        return [self.eta_d, self.eta_i, self.eta_e]

    @property
    def eta4(self):
        return [self.eta_d, self.eta_e]

    @property
    def beta_final(self):
        return self.beta_schedule[-1]


@dataclass(kw_only=True)
class FilterAndProject:
    """
    Helmholtz filter and Heaviside projection helper.
    """
    radius: float
    mesh: Any
    solver_settings: Dict = field(default_factory=lambda: {
        "linear_solver": "cg",
        "preconditioner": "hypre_amg",
    })

    _function_space: Optional[Any] = field(default=None, init=False, repr=False)

    @property
    def Vd(self):
        if self._function_space is None:
            self._function_space = fs.FunctionSpace(self.mesh, "P", 1)
        return self._function_space

    @property
    def second_radius(self):
        return self.radius

    def filter(self, density, radius: Optional[float] = None):
        """
        Solves -r^2 Laplace(rho_tilde) + rho_tilde = rho.
        """
        if radius is None:
            radius = self.radius

        rho = fs.TrialFunction(self.Vd)
        v = fs.TestFunction(self.Vd)
        a = (
            fa.Constant(radius**2)*fs.inner(fs.grad(rho), fs.grad(v))
            + rho*v
        )*fs.dx
        L = density*v*fs.dx
        rho_tilde = fa.Function(self.Vd, name="rho_tilde")

        fa.solve(a == L,
                 rho_tilde,
                 solver_parameters=self.solver_settings)

        return rho_tilde

    def project(self, rho_tilde, beta: float, eta: float):
        """
        Smooth Heaviside projection.
        """
        beta = fa.Constant(beta)
        eta = fa.Constant(eta)
        numerator = tanh(beta*eta) + tanh(beta*(rho_tilde - eta))
        denominator = tanh(beta*eta) + tanh(beta*(1.0 - eta))
        rho_bar = fa.project(numerator/denominator, self.Vd)
        rho_bar.rename("rho_bar", "projected density")
        return rho_bar

    def double_filter(self, density, beta: float, eta_min: float, eta: float):
        """
        Creates one robust physical density realization.
        """
        rho_tilde_1 = self.filter(density, self.radius)
        rho_bar_1 = self.project(rho_tilde_1, beta, eta_min)
        rho_tilde_2 = self.filter(rho_bar_1, self.second_radius)
        return self.project(rho_tilde_2, beta, eta)


@dataclass(kw_only=True)
class BowspritTopOpt(CantileverBeam2dLinear):
    """
    SIMP topology optimization solver for the bowsprit exam problem.
    """
    bowsprit_loads: BowspritLoadProperties = field(default_factory=BowspritLoadProperties)
    penalty: float = 3.0
    e_min_fraction: float = 1e-6
    filter_radius: float = 0.04
    volume_fraction: float = 0.33
    pitch_weight_alpha: float = 2.0
    elasticity_solver_settings: Dict = field(default_factory=lambda: {"linear_solver": "mumps"})
    filter_solver_settings: Dict = field(default_factory=lambda: {
        "linear_solver": "cg",
        "preconditioner": "hypre_amg",
    })

    _filter_project: Optional[FilterAndProject] = field(default=None, init=False, repr=False)
    _rho: Optional[Any] = field(default=None, init=False, repr=False)
    _Jhat: Optional[Any] = field(default=None, init=False, repr=False)
    _Vhat: Optional[Any] = field(default=None, init=False, repr=False)
    _Phat: Optional[Any] = field(default=None, init=False, repr=False)

    _history_rho_file: Optional[Any] = field(default=None, init=False, repr=False)
    _history_rho_bar_file: Optional[Any] = field(default=None, init=False, repr=False)
    _history_realization_files: Optional[Dict[float, Any]] = field(default=None, init=False, repr=False)
    _history_beta: Optional[float] = field(default=None, init=False, repr=False)
    _history_eta_values: Optional[List[float]] = field(default=None, init=False, repr=False)
    _history_eta: float = field(default=0.5, init=False, repr=False)
    _history_stride: int = field(default=10, init=False, repr=False)
    _history_iteration: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        self._create_function_spaces()
        self._mark_end_faces()
        self._filter_project = FilterAndProject(
            radius=self.filter_radius,
            mesh=self.mesh,
            solver_settings=self.filter_solver_settings,
        )
        self._rho = fa.interpolate(fa.Constant(self.volume_fraction), self._filter_project.Vd)
        self._rho.rename("rho", "design variable")

    @property
    def mesh(self):
        if self._mesh is not None:
            return self._mesh

        length = self.geometry_properties.length
        height = self.geometry_properties.height
        nx, ny = self.mesh_size
        self._mesh = fs.RectangleMesh.create(
            [fs.Point(0.0, 0.0), fs.Point(length, height)],
            [nx, ny],
            fs.CellType.Type.triangle,
        )
        return self._mesh

    def _mark_end_faces(self, tol=1e-10):
        """
        Marks clamp, F1 strip, and F2 strip.
        """
        marking_areas = self._get_boundary_area()
        marking_areas.set_all(0)

        length = self.geometry_properties.length
        height = self.geometry_properties.height
        loads = self.bowsprit_loads

        f1_center = loads.f1_x_center_frac*length
        f1_half_width = loads.f1_strip_half_frac*length
        f1_x_min = f1_center - f1_half_width
        f1_x_max = f1_center + f1_half_width
        f2_x_min = loads.f2_x_min_frac*length
        f2_x_max = loads.f2_x_max_frac*length

        class fixed_boundary(fs.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fs.near(x[0], 0.0, tol)

        class f1_boundary(fs.SubDomain):
            def inside(self, x, on_boundary):
                return (
                    on_boundary
                    and fs.near(x[1], height, tol)
                    and f1_x_min - tol <= x[0] <= f1_x_max + tol
                )

        class f2_boundary(fs.SubDomain):
            def inside(self, x, on_boundary):
                return (
                    on_boundary
                    and fs.near(x[1], height, tol)
                    and f2_x_min - tol <= x[0] <= f2_x_max + tol
                )

        fixed_boundary().mark(marking_areas, CLAMP_MARKER)
        f1_boundary().mark(marking_areas, F1_MARKER)
        f2_boundary().mark(marking_areas, F2_MARKER)

        self._boundary_applied_areas = fs.Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=marking_areas,
        )
        return marking_areas

    def _traction_forces(self):
        """
        Converts total forces to tractions on their boundary strips.
        """
        loads = self.bowsprit_loads
        length = self.geometry_properties.length
        thickness = self.geometry_properties.thickness
        f1_width = 2.0*loads.f1_strip_half_frac*length
        f2_width = (loads.f2_x_max_frac - loads.f2_x_min_frac)*length
        alpha_1 = np.deg2rad(loads.f1_angle_deg)
        alpha_2 = np.deg2rad(loads.f2_angle_deg)

        T1 = fa.Constant((
            -loads.f1_total_force*np.cos(alpha_1)/(f1_width*thickness),
            loads.f1_total_force*np.sin(alpha_1)/(f1_width*thickness),
        ))
        T2 = fa.Constant((
            -loads.f2_total_force*np.cos(alpha_2)/(f2_width*thickness),
            loads.f2_total_force*np.sin(alpha_2)/(f2_width*thickness),
        ))
        return T1, T2

    def stresses_with_simp(self, displacement, rho_bar):
        """
        Calculates stress with SIMP interpolation of Young's modulus.
        """
        epsilon = self.strains(displacement)
        E0 = self.material_properties.e_modulus
        nu = self.material_properties.poisson_ratio
        E_min = self.e_min_fraction*E0
        E = E_min + rho_bar**self.penalty*(E0 - E_min)
        mu = E/(2.0*(1.0 + nu))
        lame_lambda = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))
        dimensions = displacement.geometric_dimension()
        return lame_lambda*fs.tr(epsilon)*fs.Identity(dimensions) + 2.0*mu*epsilon

    def physical_density(self, beta: float, eta_values: List[float], eta: float):
        """
        Creates one projected physical density field.
        """
        return self._filter_project.double_filter(
            self._rho,
            beta,
            min(eta_values),
            eta,
        )

    def forward(self, beta: float, eta_values: List[float], eta: float):
        """
        Runs projection and elasticity for one robust realization.
        """
        rho_bar = self.physical_density(beta, eta_values, eta)
        displacement = self.solve_topopt(rho_bar)
        return displacement, rho_bar

    def solve_topopt(self, rho_bar, solver_settings: Optional[Dict] = None):
        """
        Solves the SIMP linear elasticity problem.
        """
        if solver_settings is None:
            solver_settings = self.elasticity_solver_settings

        u = fs.TrialFunction(self._function_space)
        v = fs.TestFunction(self._function_space)
        T1, T2 = self._traction_forces()
        ds = self._boundary_applied_areas
        thickness = self.geometry_properties.thickness

        bc = fa.DirichletBC(
            self._function_space,
            fa.Constant((0.0, 0.0)),
            self._boundary_area,
            CLAMP_MARKER,
        )
        a = thickness*fs.inner(self.stresses_with_simp(u, rho_bar), self.strains(v))*fs.dx
        L = thickness*(fs.dot(T1, v)*ds(F1_MARKER) + fs.dot(T2, v)*ds(F2_MARKER))
        displacement = fa.Function(self._function_space, name="Displacement")

        fa.solve(a == L,
                 displacement,
                 bc,
                 solver_parameters=solver_settings)

        return displacement

    def compliance(self, displacement):
        """
        Calculates external work compliance.
        """
        T1, T2 = self._traction_forces()
        thickness = self.geometry_properties.thickness
        ds = self._boundary_applied_areas
        return fa.assemble(thickness*(fs.dot(T1, displacement)*ds(F1_MARKER) + fs.dot(T2, displacement)*ds(F2_MARKER)))

    def volume_fraction_of(self, rho_bar):
        """
        Calculates volume fraction.
        """
        area = self.geometry_properties.length*self.geometry_properties.height
        return fa.assemble(rho_bar*fs.dx)/area

    def pitch_weighted_volume_fraction_of(self, rho_bar):
        """
        Calculates weighted volume fraction for pitching-inertia penalty.
        """
        length = self.geometry_properties.length
        height = self.geometry_properties.height
        x = fs.SpatialCoordinate(self.mesh)
        alpha = fa.Constant(self.pitch_weight_alpha)
        weight = fa.Constant(1.0) + alpha*(x[0]/fa.Constant(length))**2
        reference = length*height*(1.0 + self.pitch_weight_alpha/3.0)
        return fa.assemble(weight*rho_bar*fs.dx)/reference

    def volume_constraint(self, rho_bar):
        """
        Returns V - Vmax <= 0.
        """
        return self.volume_fraction_of(rho_bar) - self.volume_fraction

    def pitch_constraint(self, rho_bar):
        """
        Returns pitch-weighted volume - Vmax <= 0.
        """
        return self.pitch_weighted_volume_fraction_of(rho_bar) - self.volume_fraction

    def set_up_functionals(self, beta: float, eta_values: List[float]):
        """
        Builds reduced functionals for objective and constraints.
        """
        control = fa.Control(self._rho)
        tape = fa.get_working_tape()

        tape.clear_tape()
        total_compliance = None
        for eta in sorted(eta_values):
            displacement, _ = self.forward(beta, eta_values, eta)
            J = self.compliance(displacement)
            total_compliance = J if total_compliance is None else total_compliance + J

        objective_tape = tape.copy()
        objective_tape.optimize(controls=[control], functionals=[total_compliance])
        self._Jhat = fa.ReducedFunctional(total_compliance, control, tape=objective_tape)

        tape.clear_tape()
        rho_bar_dilated = self.physical_density(beta, eta_values, min(eta_values))
        V = self.volume_constraint(rho_bar_dilated)
        volume_tape = tape.copy()
        volume_tape.optimize(controls=[control], functionals=[V])
        self._Vhat = fa.ReducedFunctional(V, control, tape=volume_tape)

        tape.clear_tape()
        rho_bar_pitch = self.physical_density(beta, eta_values, min(eta_values))
        P = self.pitch_constraint(rho_bar_pitch)
        pitch_tape = tape.copy()
        pitch_tape.optimize(controls=[control], functionals=[P])
        self._Phat = fa.ReducedFunctional(P, control, tape=pitch_tape)

    def f(self, x):
        """
        MMA function vector [objective, volume_constraint, pitch_constraint].
        """
        self._set_density_from_petsc(x)
        objective = self._Jhat(self._rho)
        volume = self._Vhat(self._rho)
        pitch = self._Phat(self._rho)
        return np.array([float(objective), float(volume), float(pitch)])

    def g(self, x):
        """
        MMA gradient matrix.
        """
        self._set_density_from_petsc(x)
        dJ = self._Jhat.derivative().vector().get_local()
        dV = self._Vhat.derivative().vector().get_local()
        dP = self._Phat.derivative().vector().get_local()
        istart, iend = x.getOwnershipRange()
        return np.array([dJ[istart:iend], dV[istart:iend], dP[istart:iend]])

    def _set_density_from_petsc(self, x):
        rho_array = MMA_petsc.parToLocal(x)
        self._rho.vector().set_local(rho_array)
        self._rho.vector().apply("")

    def set_up_optimizer(self, x0, move: float, kmax: int):
        """
        Initializes MMA.
        """
        mma = MMA_petsc(x0, 2, f=self.f, g=self.g, plot_k=self.plot_k)
        mma.xmin[:] = 0.0
        mma.xmax[:] = 1.0
        mma.move = move
        mma.xtol = 1e-4
        mma.ftol = 1e-5
        mma.lmax = 5
        mma.kmax = kmax
        mma.kmin = 1
        return mma

    @staticmethod
    def eta_label(eta: float, eta_values: List[float]):
        """
        Gives a readable name to one robust density realization.
        """
        eta_min = min(eta_values)
        eta_max = max(eta_values)
        if np.isclose(eta, eta_min):
            return "dilated"
        if np.isclose(eta, eta_max):
            return "eroded"
        if np.isclose(eta, 0.5):
            return "nominal"
        return f"eta_{eta:.2f}".replace(".", "p")

    def set_up_history(self, output_dir: str, prefix: str, stride: int, eta_values: List[float]):
        """
        Creates PVD files for ParaView animation.
        """
        os.makedirs(output_dir, exist_ok=True)
        self._history_rho_file = fs.File(os.path.join(output_dir, f"{prefix}_rho_history.pvd"))
        self._history_rho_bar_file = fs.File(os.path.join(output_dir, f"{prefix}_rho_bar_history.pvd"))
        self._history_realization_files = {
            eta: fs.File(os.path.join(
                output_dir,
                f"{prefix}_{self.eta_label(eta, eta_values)}_rho_bar_history.pvd",
            ))
            for eta in sorted(eta_values)
        }
        self._history_stride = stride
        self._history_iteration = 0

    def plot_k(self, x):
        """
        MMA callback for writing optimization history.
        """
        if self._history_rho_file is None:
            return
        if self._history_iteration % self._history_stride != 0:
            self._history_iteration += 1
            return

        self._set_density_from_petsc(x)
        self._rho.rename("rho", "design variable")
        self._history_rho_file << (self._rho, self._history_iteration)

        fa.pause_annotation()
        try:
            rho_bar = self.physical_density(
                self._history_beta,
                self._history_eta_values,
                self._history_eta,
            )
            rho_bar.rename("rho_bar", "physical density")
            self._history_rho_bar_file << (rho_bar, self._history_iteration)

            for eta, history_file in self._history_realization_files.items():
                rho_bar_eta = self.physical_density(
                    self._history_beta,
                    self._history_eta_values,
                    eta,
                )
                rho_bar_eta.rename("rho_bar", f"{self.eta_label(eta, self._history_eta_values)} density")
                history_file << (rho_bar_eta, self._history_iteration)
        finally:
            fa.continue_annotation()

        self._history_iteration += 1

    def optimize(self,
                 eta_values: List[float],
                 beta_schedule: List[float],
                 kmax_per_stage: int,
                 move: float,
                 history_output_dir: Optional[str] = None,
                 history_prefix: str = "history",
                 history_stride: int = 10):
        """
        Runs beta-continuation topology optimization.
        """
        rho_petsc = fa.as_backend_type(self._rho.vector()).vec()
        mma = self.set_up_optimizer(rho_petsc, move, kmax_per_stage)

        if history_output_dir is not None:
            self.set_up_history(history_output_dir, history_prefix, history_stride, eta_values)

        rank = PETSc.COMM_WORLD.getRank()
        if rank == 0 and tqdm is not None:
            beta_iterator = tqdm(
                beta_schedule,
                desc=f"{history_prefix} beta stages",
                unit="stage",
            )
        else:
            beta_iterator = beta_schedule

        for beta in beta_iterator:
            if rank == 0 and tqdm is not None:
                beta_iterator.set_postfix(beta=beta, eta=sorted(eta_values))
            elif rank == 0:
                print("beta=", beta, "eta=", sorted(eta_values))
            self._history_beta = beta
            self._history_eta_values = eta_values
            self.set_up_functionals(beta, eta_values)
            mma.solve(rho_petsc)

        self._set_density_from_petsc(mma.x)
        return MMA_petsc.parToLocal(mma.x)

    def evaluate_design(self,
                        beta: float,
                        eta_values: List[float],
                        eta: float,
                        rho_array: Optional[np.ndarray] = None):
        """
        Evaluates one realization without recording on the pyadjoint tape.
        """
        if rho_array is not None:
            self._rho.vector().set_local(rho_array)
            self._rho.vector().apply("")

        fa.pause_annotation()
        try:
            displacement, rho_bar = self.forward(beta, eta_values, eta)
        finally:
            fa.continue_annotation()

        return displacement, rho_bar

    def save_design(self,
                    beta: float,
                    eta_values: List[float],
                    eta: float,
                    prefix: str,
                    output_dir: str,
                    rho_array: Optional[np.ndarray] = None):
        """
        Saves one final density and displacement realization.
        """
        os.makedirs(output_dir, exist_ok=True)
        displacement, rho_bar = self.evaluate_design(beta, eta_values, eta, rho_array)
        rho_bar.rename("rho_bar", "physical density")
        fs.File(os.path.join(output_dir, f"{prefix}_rho_bar.pvd")) << rho_bar
        fs.File(os.path.join(output_dir, f"{prefix}_u.pvd")) << displacement
        return (
            float(self.compliance(displacement)),
            float(self.volume_fraction_of(rho_bar)),
            float(self.pitch_weighted_volume_fraction_of(rho_bar)),
        )

    def save_marked_faces(self, output_dir: str, prefix: str = "bowsprit"):
        """
        Saves boundary markers for manual inspection in ParaView.
        """
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{prefix}_marked_faces.pvd")
        fs.File(filename) << self._boundary_area
        return filename


def make_run_directory(base_dir="plots"):
    """
    Creates a timestamped output folder.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def make_bowsprit(material, geometry, loads, settings: OptimizationSettings):
    """
    Creates a bowsprit optimizer from the shared run settings.
    """
    return BowspritTopOpt(
        material_properties=material,
        geometry_properties=geometry,
        loads=loads,
        mesh_size=settings.mesh_size,
        verbose=False,
        filter_radius=settings.filter_radius,
        volume_fraction=settings.volume_fraction,
        pitch_weight_alpha=settings.pitch_weight_alpha,
    )


def save_final_realizations(beam, beta, eta_values, rho_array, task_name, labels, output_root):
    results = {}
    output_dir = os.path.join(output_root, task_name)
    for eta, label in labels.items():
        result = beam.save_design(
            beta,
            eta_values,
            eta,
            f"{task_name}_{label}",
            output_dir,
            rho_array,
        )
        results[label] = result
        if PETSc.COMM_WORLD.getRank() == 0:
            print(
                f"{task_name} {label}: "
                f"J = {result[0]:.4e}, V = {result[1]:.3f}, V_pitch = {result[2]:.3f}"
            )
    return results


def print_comparison(results3, results4):
    if PETSc.COMM_WORLD.getRank() != 0:
        return

    print("\nDesign comparison")
    print(f"{'Design':24s} {'J':>12s} {'V':>8s} {'V_pitch':>10s}")
    print("-"*60)

    for label in ["dilated", "nominal", "eroded"]:
        J, V, V_pitch = results3[label]
        print(f"3-design {label:12s} {J:12.4e} {V:8.3f} {V_pitch:10.3f}")

    for label in ["dilated", "intermediate", "eroded"]:
        J, V, V_pitch = results4[label]
        print(f"2-design {label:12s} {J:12.4e} {V:8.3f} {V_pitch:10.3f}")

    difference = (
        100.0*(results4["intermediate"][0] - results3["nominal"][0])
        / results3["nominal"][0]
    )
    print(f"\nIntermediate comparison difference: {difference:+.2f} %")


if __name__ == "__main__":
    fs.set_log_level(30)

    material = MaterialProperties2d(e_modulus=70e9, poisson_ratio=0.33, density=None)
    geometry = GeometryProperties2d(length=3.80, height=0.5, thickness=0.05)
    loads = LoadCase2d()
    settings = OptimizationSettings(mesh_size=(460,60))
    history_stride = int(os.environ.get("BOWSPRIT_HISTORY_STRIDE", "10"))
    output_root = make_run_directory("plots")

    if PETSc.COMM_WORLD.getRank() == 0:
        print("Bowsprit topology optimization")
        print("Output:", output_root)

    # Task 3: optimize using dilated, nominal, and eroded realizations.
    beam3 = make_bowsprit(material, geometry, loads, settings)
    marker_file = beam3.save_marked_faces(output_root)
    if PETSc.COMM_WORLD.getRank() == 0:
        print("Marked faces:", marker_file)

    rho3 = beam3.optimize(
        eta_values=settings.eta3,
        beta_schedule=settings.beta_schedule,
        kmax_per_stage=settings.kmax,
        move=settings.move,
        history_output_dir=os.path.join(output_root, "task3"),
        history_prefix="task3",
        history_stride=history_stride,
    )
    results3 = save_final_realizations(
        beam3,
        settings.beta_final,
        settings.eta3,
        rho3,
        "task3",
        {settings.eta_d: "dilated", settings.eta_i: "nominal", settings.eta_e: "eroded"},
        output_root,
    )

    # Task 4: optimize using dilated and eroded realizations only.
    beam4 = make_bowsprit(material, geometry, loads, settings)
    rho4 = beam4.optimize(
        eta_values=settings.eta4,
        beta_schedule=settings.beta_schedule,
        kmax_per_stage=settings.kmax,
        move=settings.move,
        history_output_dir=os.path.join(output_root, "task4"),
        history_prefix="task4",
        history_stride=history_stride,
    )
    results4 = save_final_realizations(
        beam4,
        settings.beta_final,
        settings.eta4,
        rho4,
        "task4",
        {settings.eta_d: "dilated", settings.eta_e: "eroded"},
        output_root,
    )
    results4["intermediate"] = beam4.save_design(
        settings.beta_final,
        settings.eta4,
        settings.eta_i,
        "task4_intermediate",
        os.path.join(output_root, "task4"),
        rho4,
    )

    print_comparison(results3, results4)
