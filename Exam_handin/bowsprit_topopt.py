"""
Topology optimization of a bowsprit beam (2D, linear-elastic cantilever).

Two angled surface tractions on the top face. Uses SIMP density interpolation,
double-filter robust formulation (Lecture 10), pyadjoint adjoint sensitivities,
and MMA. Supports serial and MPI-parallel modes — one realization group per eta value.

Compliance objective J = ∫ T·u ds (external work, equals strain energy for
linear elasticity).

_build_dof_mapping replaces parallel.py:create_mapping, which pairs DOFs by
np.isin position rather than by coordinate and causes problems with the mapping when the
two communicators partition cells differently.

beam_configurator_2d.py is my own submission from the assignment earlier in the course.
mma.py is from the brightspace course site by Søren Madsen
parallel.py is from the brightspace course site by Søren Madsen
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import fenics as fs
from fenics import MPI
import fenics_adjoint as fa
import numpy as np
from petsc4py import PETSc
from scipy.spatial import cKDTree
from ufl import sqrt, tanh

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
from parallel import Parallel


CLAMP_MARKER = 1
F1_MARKER = 2
F2_MARKER = 3


def _make_mma(x0, ncon, f, g, plot_k, move, lmax, kmax):
    """Initializes MMA with standard topology optimization bounds and tolerances using mma.py"""
    mma = MMA_petsc(x0, ncon, f=f, g=g, plot_k=plot_k)
    mma.xmin[:] = 0.0
    mma.xmax[:] = 1.0
    mma.move = move
    mma.xtol = 1e-4
    mma.ftol = 1e-5
    mma.lmax = lmax
    mma.kmax = kmax
    mma.kmin = 1
    return mma


def _make_reduced_functional(functional, control):
    """Copies and optimizes the current tape, then wraps it in a ReducedFunctional.
    Each functional gets its own tape so derivative() only replays the relevant
    computation.
    """
    tape = fa.get_working_tape()
    t = tape.copy()
    t.optimize(controls=[control], functionals=[functional])
    return fa.ReducedFunctional(functional, control, tape=t)


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
    f1_strip_half_frac: float = 0.025
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


@dataclass(frozen=True, slots=True, kw_only=True)
class StressConstraintSettings:
    """
    Parameters for the global P-mean stress constraint (Lecture 11/12).
    sigma_y: yield stress [Pa], e.g. 250e6 for Al 6061-T6.
    beta_cap: max beta used when computing rho_bar for stress; default 4.8
              (= beta_lim/2 for R=0.04 m, ny=30, H=0.5 m — avoids artificial
              stress concentrations at sharp Heaviside transitions).
    """
    sigma_y: float
    epsilon_relaxation: float = 0.2
    p_stress_schedule: Tuple[int, ...] = (2, 2, 4, 4, 8, 8, 16, 32, 64, 128, 200, 300, 300)
    alpha: float = 1.0
    beta_cap: Optional[float] = 4.8


@dataclass(frozen=True, slots=True)
class OptimizationSettings:
    """
    Tunable parameters for the optimization run.
    """
    mesh_size: Tuple[int, int] = (230, 30)
    filter_radius: float = 0.04
    volume_fraction: float = 0.25
    pitch_weight_alpha: float = 2.5
    beta_schedule: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    lmax: int = 5
    kmax: int = 20
    move: float = 0.1
    eta_d: float = 0.45
    eta_i: float = 0.5
    eta_e: float = 0.55
    stress: Optional[StressConstraintSettings] = None

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
    _filter_space: Optional[Any] = field(default=None, init=False, repr=False)

    @property
    def Vd(self):
        if self._function_space is None:
            self._function_space = fs.FunctionSpace(self.mesh, "DG", 0)
        return self._function_space

    @property
    def Vf(self):
        # CG1 for filter intermediates: DG0 grad = 0 inside cells, so the
        # Helmholtz diffusion term vanishes if Vd is used here.
        if self._filter_space is None:
            self._filter_space = fs.FunctionSpace(self.mesh, "P", 1)
        return self._filter_space

    @property
    def second_radius(self):
        return self.radius

    def filter(self, density, radius: Optional[float] = None):
        """
        Solves -r^2 Laplace(rho_tilde) + rho_tilde = rho in CG1 (Lecture 10 PDE filter).
        CG1 required: DG0 gradients vanish inside cells, killing the diffusion term.
        """
        if radius is None:
            radius = self.radius

        rho = fs.TrialFunction(self.Vf)
        v = fs.TestFunction(self.Vf)
        a = (
            fa.Constant(radius**2)*fs.inner(fs.grad(rho), fs.grad(v))
            + rho*v
        )*fs.dx
        L = density*v*fs.dx
        rho_tilde = fa.Function(self.Vf, name="rho_tilde")

        fa.solve(a == L,
                 rho_tilde,
                 solver_parameters=self.solver_settings)

        return rho_tilde

    def project(self, rho_tilde, beta: float, eta: float):
        """
        Smooth Heaviside projection via tanh regularization (Lecture 10).
        beta controls sharpness; eta is the projection threshold.
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
        One robust realization via the double-filter pipeline (Lecture 10):
        rho -> filter -> project(eta_min) -> filter -> project(eta_k).
        eta_min is the dilated threshold (0.45); eta_k is realization-specific.
        """
        rho_tilde_1 = self.filter(density, self.radius)
        rho_bar_1 = self.project(rho_tilde_1, beta, eta_min)
        rho_tilde_2 = self.filter(rho_bar_1, self.second_radius)
        return self.project(rho_tilde_2, beta, eta)


@dataclass(kw_only=True)
class BowspritTopOpt(CantileverBeam2dLinear):
    """
    SIMP topology optimization solver for the bowsprit problem.
    """
    bowsprit_loads: BowspritLoadProperties = field(default_factory=BowspritLoadProperties)
    penalty: float = 3.0
    e_min_fraction: float = 1e-6
    filter_radius: float = 0.04
    volume_fraction: float = 0.33
    pitch_weight_alpha: float = 2.0
    elasticity_solver_settings: Dict = field(default_factory=lambda: {
        "linear_solver": "mumps",
    })
    filter_solver_settings: Dict = field(default_factory=lambda: {
        "linear_solver": "gmres",
        "preconditioner": "hypre_amg",
    })

    #Initialization of attributes and handling of initialization
    _filter_project: Optional[FilterAndProject] = field(default=None, init=False, repr=False)
    _rho: Optional[Any] = field(default=None, init=False, repr=False)
    _Jhat: Optional[Any] = field(default=None, init=False, repr=False)
    _Vhat: Optional[Any] = field(default=None, init=False, repr=False)
    _Phat: Optional[Any] = field(default=None, init=False, repr=False)
    _Shat: List[Any] = field(default_factory=list, init=False, repr=False)

    _history_rho_file: Optional[Any] = field(default=None, init=False, repr=False)
    _history_rho_bar_file: Optional[Any] = field(default=None, init=False, repr=False)
    _history_realization_files: Optional[Dict[float, Any]] = field(default=None, init=False, repr=False)
    _history_beta: Optional[float] = field(default=None, init=False, repr=False)
    _history_eta_values: Optional[List[float]] = field(default=None, init=False, repr=False)
    _history_eta: float = field(default=0.5, init=False, repr=False)
    _history_stride: int = field(default=10, init=False, repr=False)
    _history_iteration: int = field(default=0, init=False, repr=False)

    #Post init creates the model setup after the class is instantiated
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

        #converts using -cos and sin as the angle is defined on the left side
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
        Stress with SIMP stiffness interpolation (Lecture 6 p. 9):
        E(rho_bar) = E_min + rho_bar^p * (E0 - E_min).
        Plane-stress Lamé: lambda_eff = E*nu/(1-nu^2) as it is a thin bowsprit (lecture 3 p. 7-9).
        """
        epsilon = self.strains(displacement)
        E0 = self.material_properties.e_modulus
        nu = self.material_properties.poisson_ratio
        E_min = self.e_min_fraction*E0
        E = E_min + rho_bar**self.penalty*(E0 - E_min)
        mu = E/(2.0*(1.0 + nu))
        lame_lambda = E*nu/(1.0 - nu**2)
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
        Solves the SIMP linear elasticity problem (Lecture 3 p. 7-9 plane-stress BVP):
        a(u,v) = t * ∫ sigma(u):eps(v) dx
        L(v)   = t * ∫ T1.v ds(F1) + T2.v ds(F2)
        MUMPS direct solver: SIMP E_min/E0=1e-6 causes ill-conditioning
            that can be problematic with other solvers at high beta values (e.g. gmres diverged)
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
        Calculates external work compliance J = ∫ T.u ds.
        Equals strain energy for linear elasticity; minimizing J maximizes stiffness.
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
        Pitch-weighted volume fraction with w(x) = 1 + alpha*(x/L)^2.
        Evaluated on the dilated realization — if dilated satisfies it,
        all realizations do.
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

    def stresses_base_material(self, displacement):
        """
        Plane-stress tensor computed with base-material E (not SIMP), for stress constraints.
        Lecture 11 slide 7: stress in constraint uses E0, not E(rho_bar), so that
        f_sigma(rho_bar) fully accounts for the density interpolation.
        Plane-stress Lame: lambda_eff = E*nu/(1-nu^2), mu = E/(2*(1+nu)).
        """
        epsilon = self.strains(displacement)
        E0 = self.material_properties.e_modulus
        nu = self.material_properties.poisson_ratio
        mu = E0 / (2.0 * (1.0 + nu))
        lame_lambda = E0 * nu / (1.0 - nu**2)
        d = displacement.geometric_dimension()
        return lame_lambda * fs.tr(epsilon) * fs.Identity(d) + 2.0 * mu * epsilon

    def stress_measure(self, displacement, rho_bar, sigma_y, epsilon_relaxation=0.2):
        """
        Local stress measure σ_m = f_σ(ρ̄)·√(σ²_vM + σ²_min), Lecture 11 slide 7.
        f_σ(ρ̄) = ρ̄ / (ε(1-ρ̄) + ρ̄)  relaxes singularity in void regions.
        σ_vM uses base-material E; σ_min = 1e-4·σ_y prevents 0/0 in voids.
        """
        sigma = self.stresses_base_material(displacement)
        d = displacement.geometric_dimension()
        mean_stress = fs.tr(sigma) / 3.0        # (σ_11+σ_22)/3 = full 3D mean (σ_33=0)
        s = sigma - mean_stress * fs.Identity(d)
        # s_33 = -mean_stress (plane stress: σ_33=0 but s_33≠0)
        sigma_vm2 = 1.5 * (fs.inner(s, s) + mean_stress**2)
        sigma_min_sq = (1e-4 * sigma_y) ** 2
        eps = fa.Constant(epsilon_relaxation)
        f_sigma = rho_bar / (eps * (1.0 - rho_bar) + rho_bar)
        return f_sigma * sqrt(sigma_vm2 + sigma_min_sq)

    def p_mean_stress_constraint(self, displacement, rho_bar, p, sigma_y,
                                  alpha=1.0, epsilon_relaxation=0.2):
        """
        Returns σ_c - 1 where σ_c = ((1/|Ω|) ∫ (σ_m/(α·σ_y))^p dΩ)^(1/p).
        Constraint σ_c - 1 ≤ 0, Lecture 11 slide 12 or Lecture 12 slide 4.
        """
        sm = self.stress_measure(displacement, rho_bar, sigma_y, epsilon_relaxation)
        area = self.geometry_properties.length * self.geometry_properties.height
        sigma_ref = alpha * sigma_y
        p_const = fa.Constant(float(p))
        Ip = fa.assemble((sm / sigma_ref) ** p_const * fs.dx) / area
        return Ip ** (1.0 / float(p)) - 1.0

    def set_up_functionals(self, beta: float, eta_values: List[float],
                           p_stress: Optional[int] = None,
                           stress_settings: Optional[StressConstraintSettings] = None):
        """
        Builds reduced functionals for objective and constraints.
        Each functional gets its own optimized tape so derivative() only replays
        the relevant computation. V and P are evaluated on the dilated realization.
        When stress_settings is provided, also builds one stress functional per eta.
        """
        control = fa.Control(self._rho)
        tape = fa.get_working_tape()

        # Objective: sum compliance across all realizations.
        tape.clear_tape()
        total_compliance = None
        for eta in sorted(eta_values):
            displacement, _ = self.forward(beta, eta_values, eta)
            J = self.compliance(displacement)
            total_compliance = J if total_compliance is None else total_compliance + J

        self._Jhat = _make_reduced_functional(total_compliance, control)

        # Volume constraint on dilated realization.
        tape.clear_tape()
        rho_bar_dilated = self.physical_density(beta, eta_values, min(eta_values))
        V = self.volume_constraint(rho_bar_dilated)
        self._Vhat = _make_reduced_functional(V, control)

        # Pitch-weighted volume constraint on dilated realization.
        tape.clear_tape()
        rho_bar_pitch = self.physical_density(beta, eta_values, min(eta_values))
        P = self.pitch_constraint(rho_bar_pitch)
        self._Phat = _make_reduced_functional(P, control)

        # Stress constraint per realization: Lecture 11 slide 7 + Lecture 12 slide 4.
        # beta_s capped at beta_cap to avoid artificial concentrations (Lecture 11 slide 9).
        self._Shat = []
        if stress_settings is not None and p_stress is not None:
            s = stress_settings
            beta_s = min(beta, s.beta_cap) if s.beta_cap is not None else beta
            for eta in sorted(eta_values):
                tape.clear_tape()
                displacement_s, rho_bar_s = self.forward(beta_s, eta_values, eta)
                S = self.p_mean_stress_constraint(
                    displacement_s, rho_bar_s, p_stress, s.sigma_y, s.alpha, s.epsilon_relaxation
                )
                self._Shat.append(_make_reduced_functional(S, control))

    def f(self, x):
        """
        MMA function vector [J, V, P, S_eta0, S_eta1, ...].
        """
        self._set_density_from_petsc(x)
        result = [
            float(self._Jhat(self._rho)),   #compliance constraint 
            float(self._Vhat(self._rho)),   #volume constraint
            float(self._Phat(self._rho)),   #pitch constraint (inertia)
        ]
        for Shat in self._Shat:
            result.append(float(Shat(self._rho)))   #stress constraint
        return np.array(result)

    def g(self, x):
        """
        MMA gradient matrix [dJ, dV, dP, dS_eta0, dS_eta1, ...].
        """
        self._set_density_from_petsc(x)
        istart, iend = x.getOwnershipRange()
        rows = [
            self._Jhat.derivative().vector().get_local()[istart:iend],  #compliance constraint
            self._Vhat.derivative().vector().get_local()[istart:iend],  #volume constraint
            self._Phat.derivative().vector().get_local()[istart:iend],  #pitch constraint (inertia)
        ]
        for Shat in self._Shat:
            rows.append(Shat.derivative().vector().get_local()[istart:iend]) #stress constraint
        return np.array(rows)

    def _set_density_from_petsc(self, x):
        rho_array = MMA_petsc.parToLocal(x)
        lo, hi = self._rho.vector().local_range()
        self._rho.vector().set_local(rho_array[lo:hi])
        self._rho.vector().apply("")

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
                 lmax_per_stage: int,
                 kmax_per_stage: int,
                 move: float,
                 history_output_dir: Optional[str] = None,
                 history_prefix: str = "history",
                 history_stride: int = 10,
                 stress_settings: Optional[StressConstraintSettings] = None):
        """
        Beta-continuation topology optimization (Lecture 10): starts with a soft
        Heaviside (low beta) and ramps up each stage to drive the design binary.
        Pass stress_settings to activate per-realization P-mean stress constraints.
        """
        ncon = 2 + len(eta_values) if stress_settings is not None else 2
        rho_petsc = fa.as_backend_type(self._rho.vector()).vec()
        mma = _make_mma(rho_petsc, ncon, self.f, self.g, self.plot_k, move,
                        lmax=lmax_per_stage, kmax=kmax_per_stage)

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

        for stage_idx, beta in enumerate(beta_iterator):
            p_stress = None
            if stress_settings is not None:
                sched = stress_settings.p_stress_schedule
                p_stress = sched[min(stage_idx, len(sched) - 1)]
            if rank == 0 and tqdm is not None:
                beta_iterator.set_postfix(beta=beta, eta=sorted(eta_values), p=p_stress)
            elif rank == 0:
                print("beta=", beta, "eta=", sorted(eta_values), "p_stress=", p_stress)
            self._history_beta = beta
            self._history_eta_values = eta_values
            self.set_up_functionals(beta, eta_values, p_stress, stress_settings)
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
            lo, hi = self._rho.vector().local_range()
            self._rho.vector().set_local(rho_array[lo:hi])
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
                    rho_array: Optional[np.ndarray] = None,
                    stress_settings: Optional["StressConstraintSettings"] = None):
        """
        Saves one final density and displacement realization.
        Returns (J, V, V_pitch, sigma_max_Pa) where sigma_max_Pa is None
        when stress_settings is not provided.
        """
        os.makedirs(output_dir, exist_ok=True)
        displacement, rho_bar = self.evaluate_design(beta, eta_values, eta, rho_array)
        rho_bar.rename("rho_bar", "physical density")
        fs.File(os.path.join(output_dir, f"{prefix}_rho_bar.pvd")) << rho_bar
        fs.File(os.path.join(output_dir, f"{prefix}_u.pvd")) << displacement

        sigma_max = None
        if stress_settings is not None:
            fa.pause_annotation()
            try:
                s = stress_settings
                sm_expr = self.stress_measure(
                    displacement, rho_bar, s.sigma_y, s.epsilon_relaxation
                )
                sigma_m = fa.project(sm_expr, fs.FunctionSpace(self.mesh, "DG", 0))
                sigma_m.rename("sigma_m", "stress measure")
                fs.File(os.path.join(output_dir, f"{prefix}_sigma_m.pvd")) << sigma_m
                sigma_max = float(sigma_m.vector().max())
            finally:
                fa.continue_annotation()

        return (
            float(self.compliance(displacement)),
            float(self.volume_fraction_of(rho_bar)),
            float(self.pitch_weighted_volume_fraction_of(rho_bar)),
            sigma_max,
        )

    def save_marked_faces(self, output_dir: str, prefix: str = "bowsprit"):
        """
        Saves boundary markers for manual inspection in ParaView.
        """
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{prefix}_marked_faces.pvd")
        fs.File(filename) << self._boundary_area
        return filename


@dataclass(kw_only=True)
class _GroupBowspritTopOpt(BowspritTopOpt):
    """BowspritTopOpt that accepts a pre-created mesh for use on a group communicator."""
    group_mesh: Optional[Any] = field(default=None, repr=False)

    def __post_init__(self):
        if self.group_mesh is not None:
            self._mesh = self.group_mesh
        super().__post_init__()


@dataclass(kw_only=True)
class ParallelBowspritOptimizer:
    """
    Runs one BowspritTopOpt per eta realization on separate MPI communicator groups
    so all realizations solve their FEM simultaneously (Lecture 8).

    Launch: mpiexec -n N python bowsprit_topopt.py
    N must be divisible by len(eta_values). For three realizations N=6 is used
    (3 groups x 2 cores); for two realizations, 2 groups x 3 cores as my ryzen 7 4800H has 8c/16t.
    """
    material_properties: MaterialProperties2d
    geometry_properties: GeometryProperties2d
    loads: LoadCase2d = field(default_factory=LoadCase2d)
    bowsprit_loads: BowspritLoadProperties = field(default_factory=BowspritLoadProperties)
    settings: OptimizationSettings = field(default_factory=OptimizationSettings)
    eta_values: List[float] = field(default_factory=list)

    _parallel: Optional[Any] = field(default=None, init=False, repr=False)
    _beam_world: Optional[BowspritTopOpt] = field(default=None, init=False, repr=False)
    _beam_group: Optional[_GroupBowspritTopOpt] = field(default=None, init=False, repr=False)
    _rho_global: Optional[Any] = field(default=None, init=False, repr=False)
    _Jhat_group: Optional[Any] = field(default=None, init=False, repr=False)
    _Vhat_group: Optional[Any] = field(default=None, init=False, repr=False)
    _Phat_group: Optional[Any] = field(default=None, init=False, repr=False)
    _Shat_group: Optional[Any] = field(default=None, init=False, repr=False)
    _current_beta: float = field(default=1.0, init=False, repr=False)
    _my_eta: float = field(default=0.5, init=False, repr=False)
    _history_rho_file: Optional[Any] = field(default=None, init=False, repr=False)
    _history_rho_bar_file: Optional[Any] = field(default=None, init=False, repr=False)
    _history_stride: int = field(default=10, init=False, repr=False)
    _history_iteration: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        ngroups = len(self.eta_values)
        eta_sorted = sorted(self.eta_values)

        # split world communicator into ngroups groups
        self._parallel = Parallel(MPI.comm_world, ngroups)
        self._my_eta = eta_sorted[self._parallel.group]

        kwargs = self._beam_kwargs()

        # world-comm beam for MMA variable space and final design evaluation
        self._beam_world = BowspritTopOpt(**kwargs)

        # group mesh created identically to BowspritTopOpt.mesh but on the group
        # communicator. RectangleMesh.create returns the fenics_adjoint-overloaded
        # Python Mesh subclass; reading from XDMF returns the raw cpp binding and
        # causes an '_ad_will_add_as_dependency' AttributeError inside fa.solve.
        length = self.geometry_properties.length
        height = self.geometry_properties.height
        nx, ny = self.settings.mesh_size
        group_mesh = fs.RectangleMesh.create(
            self._parallel.group_comm,
            [fs.Point(0.0, 0.0), fs.Point(length, height)],
            [nx, ny],
            fs.CellType.Type.triangle,
        )

        self._beam_group = _GroupBowspritTopOpt(**kwargs, group_mesh=group_mesh)

        # global rho for MMA on world comm (DG0 to match group space)
        Vd_global = fs.FunctionSpace(self._beam_world.mesh, "DG", 0)
        self._rho_global = fa.interpolate(
            fa.Constant(self.settings.volume_fraction),
            Vd_global,
        )
        self._rho_global.rename("rho", "design variable")

        # DOF mapping between world and group function spaces (see _build_dof_mapping).
        self._build_dof_mapping(Vd_global, self._beam_group._filter_project.Vd)

    def _beam_kwargs(self) -> Dict:
        return dict(
            material_properties=self.material_properties,
            geometry_properties=self.geometry_properties,
            loads=self.loads,
            bowsprit_loads=self.bowsprit_loads,
            mesh_size=self.settings.mesh_size,
            verbose=False,
            filter_radius=self.settings.filter_radius,
            volume_fraction=self.settings.volume_fraction,
            pitch_weight_alpha=self.settings.pitch_weight_alpha,
        )

    def _build_dof_mapping(self, Vd_global, Vd_group):
        """
        Builds the DOF mapping between world and group DG0 spaces.

        Replaces parallel.py:create_mapping, which pairs DOFs by their position in
        coor_all (np.isin output order) rather than by physical coordinate. When the
        two communicators partition cells differently the orderings diverge and the
        mapping is wrong. A nearest-neighbour cKDTree lookup fixes this: each
        group-local DOF is matched to the world DOF at the same coordinate.
        """
        par = self._parallel
        par.Vglobal = Vd_global
        par.Vgroup  = Vd_group

        # All world DOF coordinates and indices gathered from all processes.
        coor_loc = np.array(Vd_global.tabulate_dof_coordinates())
        dof_loc  = np.array(Vd_global.dofmap().dofs())
        coor_all = np.concatenate(MPI.comm_world.allgather(coor_loc))
        dof_all  = np.concatenate(MPI.comm_world.allgather(dof_loc))
        par.global_dofs = dof_loc  # local world DOFs for this process

        # Group-local DOF coordinates and indices.
        coor_group = np.array(Vd_group.tabulate_dof_coordinates())
        dof_group  = np.array(Vd_group.dofmap().dofs())

        # For each group-local DOF find the world DOF at the same coordinate.
        tree = cKDTree(coor_all)
        dists, idxs = tree.query(coor_group, k=1)
        assert (dists < 1e-8).all(), "Group DOF has no matching world DOF"

        # global2group[i] = world global DOF for i-th group-local DOF.
        par.global2group = list(dof_all[idxs])

        # group2global[world_DOF] = group global DOF at the same location.
        g2g = np.zeros(Vd_global.dim(), dtype=int)
        for i, idx in enumerate(idxs):
            g2g[dof_all[idx]] = dof_group[i]
        par.group2global = sum(par.group_comm.allgather(g2g))

    def set_up_functionals(self, beta: float,
                           p_stress: Optional[int] = None,
                           stress_settings: Optional[StressConstraintSettings] = None):
        """
        Builds one ReducedFunctional per functional per group.
        Each group records its own eta realization on a separate tape.
        All groups record V and P on the dilated projection; group-0's value is
        used for the constraint since rho is shared across groups.
        When stress_settings is provided, each group records its own stress functional.
        """
        control = fa.Control(self._beam_group._rho)
        tape = fa.get_working_tape()

        tape.clear_tape()
        displacement, _ = self._beam_group.forward(beta, self.eta_values, self._my_eta)
        J = self._beam_group.compliance(displacement)
        self._Jhat_group = _make_reduced_functional(J, control)

        tape.clear_tape()
        rho_bar_d = self._beam_group.physical_density(beta, self.eta_values, min(self.eta_values))
        V = self._beam_group.volume_constraint(rho_bar_d)
        self._Vhat_group = _make_reduced_functional(V, control)

        tape.clear_tape()
        rho_bar_d2 = self._beam_group.physical_density(beta, self.eta_values, min(self.eta_values))
        P = self._beam_group.pitch_constraint(rho_bar_d2)
        self._Phat_group = _make_reduced_functional(P, control)

        self._Shat_group = None
        if stress_settings is not None and p_stress is not None:
            s = stress_settings
            beta_s = min(beta, s.beta_cap) if s.beta_cap is not None else beta
            tape.clear_tape()
            displacement_s, rho_bar_s = self._beam_group.forward(beta_s, self.eta_values, self._my_eta)
            S = self._beam_group.p_mean_stress_constraint(
                displacement_s, rho_bar_s, p_stress, s.sigma_y, s.alpha, s.epsilon_relaxation
            )
            self._Shat_group = _make_reduced_functional(S, control)

    def _sync_rho(self, x):
        """Transfer MMA PETSc vector -> global rho -> group rho."""
        rho_arr = MMA_petsc.parToLocal(x)
        lo, hi = self._rho_global.vector().local_range()
        self._rho_global.vector().set_local(rho_arr[lo:hi])
        self._rho_global.vector().apply("")
        self._parallel.fglobal2group(self._rho_global, self._beam_group._rho)

    def f(self, x):
        """
        MMA function vector [J, V, P, S_group0, S_group1, ...].
        Objective sums across groups; V and P from group 0 (dilated);
        each group contributes one stress value gathered in group order.
        """
        self._sync_rho(x)
        Ji = float(self._Jhat_group(self._beam_group._rho))
        Vi = float(self._Vhat_group(self._beam_group._rho))
        Pi = float(self._Phat_group(self._beam_group._rho))
        Js = self._parallel.sgroup2global(Ji)
        Vs = self._parallel.sgroup2global(Vi)
        Ps = self._parallel.sgroup2global(Pi)
        result = [float(np.sum(Js)), float(Vs[0]), float(Ps[0])]
        if self._Shat_group is not None:
            Si = float(self._Shat_group(self._beam_group._rho))
            Ss = self._parallel.sgroup2global(Si)
            result.extend(float(s) for s in Ss)
        return np.array(result)

    def g(self, x):
        """
        MMA gradient matrix [dJ, dV, dP, dS_group0, dS_group1, ...].
        dJ sums across groups; dV, dP from group 0; each dS_k from group k.
        """
        self._sync_rho(x)
        dJs = self._parallel.vgroup2global(self._Jhat_group.derivative().vector())
        dVs = self._parallel.vgroup2global(self._Vhat_group.derivative().vector())
        dPs = self._parallel.vgroup2global(self._Phat_group.derivative().vector())
        rows = [np.sum(dJs, axis=0), dVs[0], dPs[0]]
        if self._Shat_group is not None:
            dSs = self._parallel.vgroup2global(self._Shat_group.derivative().vector())
            rows.extend(dSs)
        return np.array(rows)

    def set_up_history(self, output_dir: str, prefix: str, stride: int):
        os.makedirs(output_dir, exist_ok=True)
        self._history_rho_file = fs.File(os.path.join(output_dir, f"{prefix}_rho_history.pvd"))
        self._history_rho_bar_file = fs.File(os.path.join(output_dir, f"{prefix}_rho_bar_history.pvd"))
        self._history_stride = stride
        self._history_iteration = 0

    def plot_k(self, x):
        """MMA callback: write rho and nominal rho_bar to history files."""
        if self._history_rho_file is None:
            return
        if self._history_iteration % self._history_stride != 0:
            self._history_iteration += 1
            return
        rho_arr = MMA_petsc.parToLocal(x)
        lo, hi = self._beam_world._rho.vector().local_range()
        self._beam_world._rho.vector().set_local(rho_arr[lo:hi])
        self._beam_world._rho.vector().apply("")
        self._beam_world._rho.rename("rho", "design variable")
        self._history_rho_file << (self._beam_world._rho, self._history_iteration)
        fa.pause_annotation()
        try:
            rho_bar = self._beam_world.physical_density(
                self._current_beta, self.eta_values, 0.5,
            )
            rho_bar.rename("rho_bar", "physical density")
            self._history_rho_bar_file << (rho_bar, self._history_iteration)
        finally:
            fa.continue_annotation()
        self._history_iteration += 1

    def optimize(self,
                 beta_schedule: List[float],
                 lmax_per_stage: int,
                 kmax_per_stage: int,
                 move: float,
                 history_output_dir: Optional[str] = None,
                 history_prefix: str = "history",
                 history_stride: int = 10,
                 stress_settings: Optional[StressConstraintSettings] = None):
        """Beta-continuation topology optimization across parallel realization groups (Lecture 10)."""
        ncon = 2 + len(self.eta_values) if stress_settings is not None else 2
        rho_petsc = fa.as_backend_type(self._rho_global.vector()).vec()
        mma = _make_mma(rho_petsc, ncon, self.f, self.g, self.plot_k, move,
                        lmax=lmax_per_stage, kmax=kmax_per_stage)

        if history_output_dir is not None:
            self.set_up_history(history_output_dir, history_prefix, history_stride)

        rank = PETSc.COMM_WORLD.getRank()
        if rank == 0 and tqdm is not None:
            beta_iterator = tqdm(
                beta_schedule,
                desc=f"{history_prefix} beta stages",
                unit="stage",
            )
        else:
            beta_iterator = beta_schedule

        for stage_idx, beta in enumerate(beta_iterator):
            p_stress = None
            if stress_settings is not None:
                sched = stress_settings.p_stress_schedule
                p_stress = sched[min(stage_idx, len(sched) - 1)]
            if rank == 0 and tqdm is not None:
                beta_iterator.set_postfix(beta=beta, eta=self._my_eta, p=p_stress)
            elif rank == 0:
                print("beta=", beta, "group eta=", self._my_eta, "p_stress=", p_stress)
            self._current_beta = beta
            self.set_up_functionals(beta, p_stress, stress_settings)
            mma.solve(rho_petsc)

        rho_arr = MMA_petsc.parToLocal(mma.x)
        lo, hi = self._rho_global.vector().local_range()
        self._rho_global.vector().set_local(rho_arr[lo:hi])
        self._rho_global.vector().apply("")
        return rho_arr

    def save_design(self,
                    beta: float,
                    eta_values: List[float],
                    eta: float,
                    prefix: str,
                    output_dir: str,
                    rho_array: Optional[np.ndarray] = None,
                    stress_settings: Optional[StressConstraintSettings] = None):
        """Delegates final design evaluation to the world-comm beam."""
        return self._beam_world.save_design(
            beta, eta_values, eta, prefix, output_dir, rho_array, stress_settings
        )

    def save_marked_faces(self, output_dir: str, prefix: str = "bowsprit"):
        return self._beam_world.save_marked_faces(output_dir, prefix)


def make_run_directory(base_dir="plots"):
    """
    Creates a timestamped output folder.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_final_realizations(beam, beta, eta_values, rho_array, task_name, labels, output_root,
                            stress_settings=None):
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
            stress_settings,
        )
        results[label] = result
        if PETSc.COMM_WORLD.getRank() == 0:
            J, V, V_pitch, sigma_max = result
            stress_str = f", σ_max = {sigma_max/1e6:.1f} MPa" if sigma_max is not None else ""
            print(
                f"{task_name} {label}: "
                f"J = {J:.4e}, V = {V:.3f}, V_pitch = {V_pitch:.3f}{stress_str}"
            )
    return results


def print_comparison(results3, results4):
    if PETSc.COMM_WORLD.getRank() != 0:
        return

    has_stress = results3["dilated"][3] is not None
    header = f"{'Design':24s} {'J':>12s} {'V':>8s} {'V_pitch':>10s}"
    if has_stress:
        header += f" {'σ_max [MPa]':>14s}"
    print("\nDesign comparison")
    print(header)
    print("-" * (74 if has_stress else 60))

    for label in ["dilated", "nominal", "eroded"]:
        J, V, V_pitch, sigma_max = results3[label]
        line = f"3-design {label:12s} {J:12.4e} {V:8.3f} {V_pitch:10.3f}"
        if has_stress:
            line += f" {sigma_max/1e6:14.1f}"
        print(line)

    for label in ["dilated", "intermediate", "eroded"]:
        J, V, V_pitch, sigma_max = results4[label]
        line = f"2-design {label:12s} {J:12.4e} {V:8.3f} {V_pitch:10.3f}"
        if has_stress:
            line += f" {sigma_max/1e6:14.1f}" if sigma_max is not None else f" {'N/A':>14s}"
        print(line)

    difference = (
        100.0*(results4["intermediate"][0] - results3["nominal"][0])
        / results3["nominal"][0]
    )
    print(f"\nIntermediate comparison difference: {difference:+.2f} %")


if __name__ == "__main__":
    # Run with: mpiexec -n 6 python bowsprit_topopt.py
    #  processes: task3 uses 3 groups x 2 cores, task4 uses 2 groups x 3 cores.
    # It takes in the ballpark of 6-8 hours for the simulation due to the fine mesh
    fs.set_log_level(30)

    material = MaterialProperties2d(e_modulus=70e9, poisson_ratio=0.33, density=None)
    geometry = GeometryProperties2d(length=3.80, height=0.5, thickness=0.05)
    loads = LoadCase2d()
    beta_schedule = (1, 2, 3, 4, 5, 8, 12, 16, 24, 36, 72, 108, 144, 288, 432)
    # Stress constraint: Al 6061-T6 yield stress 250 MPa so value way below is chosen 
    #   to account for sudden dynamical loading
    # beta_cap=9.6 ≈ beta_lim/2 for R=0.04 m, ny=60, H=0.5 m (see Lecture 11 slide 9).
    # p_stress_schedule length matches beta_schedule (15 stages).
    stress = StressConstraintSettings(
        sigma_y=40e6,
        p_stress_schedule=(2, 2, 4, 4, 8, 8, 16, 32, 64, 128, 200, 300, 300, 400, 400),
        beta_cap=9.6,  # 2R/l_e = 2*0.04/(0.5/60) for ny=60 mesh
    )
    settings = OptimizationSettings(
        filter_radius=0.04, mesh_size=(460, 60), volume_fraction=0.25, pitch_weight_alpha=2.5, lmax=15, kmax=30, move=0.2, beta_schedule=beta_schedule, stress=stress
    )
    history_stride = int(os.environ.get("BOWSPRIT_HISTORY_STRIDE", "10"))
    output_root = make_run_directory("plots")

    if PETSc.COMM_WORLD.getRank() == 0:
        print("Bowsprit topology optimization")
        print("Output:", output_root)

    # Task 3: optimize using dilated, nominal, and eroded realizations in parallel.
    opt3 = ParallelBowspritOptimizer(
        material_properties=material,
        geometry_properties=geometry,
        loads=loads,
        settings=settings,
        eta_values=list(settings.eta3),
    )
    marker_file = opt3.save_marked_faces(output_root)
    if PETSc.COMM_WORLD.getRank() == 0:
        print("Marked faces:", marker_file)

    rho3 = opt3.optimize(
        beta_schedule=settings.beta_schedule,
        lmax_per_stage=settings.lmax,
        kmax_per_stage=settings.kmax,
        move=settings.move,
        history_output_dir=os.path.join(output_root, "task3"),
        history_prefix="task3",
        history_stride=history_stride,
        stress_settings=settings.stress,
    )
    results3 = save_final_realizations(
        opt3,
        settings.beta_final,
        settings.eta3,
        rho3,
        "task3",
        {settings.eta_d: "dilated", settings.eta_i: "nominal", settings.eta_e: "eroded"},
        output_root,
        stress_settings=settings.stress,
    )

    # Task 4: optimize using dilated and eroded realizations in parallel.
    opt4 = ParallelBowspritOptimizer(
        material_properties=material,
        geometry_properties=geometry,
        loads=loads,
        settings=settings,
        eta_values=list(settings.eta4),
    )
    rho4 = opt4.optimize(
        beta_schedule=settings.beta_schedule,
        lmax_per_stage=settings.lmax,
        kmax_per_stage=settings.kmax,
        move=settings.move,
        history_output_dir=os.path.join(output_root, "task4"),
        history_prefix="task4",
        history_stride=history_stride,
        stress_settings=settings.stress,
    )
    results4 = save_final_realizations(
        opt4,
        settings.beta_final,
        settings.eta4,
        rho4,
        "task4",
        {settings.eta_d: "dilated", settings.eta_e: "eroded"},
        output_root,
        stress_settings=settings.stress,
    )
    results4["intermediate"] = opt4.save_design(
        settings.beta_final,
        settings.eta4,
        settings.eta_i,
        "task4_intermediate",
        os.path.join(output_root, "task4"),
        rho4,
        settings.stress,
    )

    print_comparison(results3, results4)

    if PETSc.COMM_WORLD.getRank() == 0:
        vf = settings.volume_fraction
        tol = 0.005

        sigma_y = settings.stress.sigma_y if settings.stress is not None else None

        def _constraint_status(V, Vp, sm, vf, tol, sigma_y):
            statuses = []
            for name, val in [("vol", V), ("pitch", Vp)]:
                if val > vf + tol:
                    statuses.append(f"{name}:VIOLATED({val:.3f}>{vf:.3f})")
                elif abs(val - vf) <= tol:
                    statuses.append(f"{name}:binding")
                else:
                    statuses.append(f"{name}:slack({val:.3f})")
            if sigma_y is not None and sm is not None:
                ratio = sm / sigma_y
                if ratio > 1.0 + tol:
                    statuses.append(f"stress:VIOLATED(σ_max={sm/1e6:.1f}>{sigma_y/1e6:.0f}MPa)")
                elif abs(ratio - 1.0) <= tol:
                    statuses.append(f"stress:binding(σ_max={sm/1e6:.1f}MPa)")
                else:
                    statuses.append(f"stress:slack(σ_max={sm/1e6:.1f}MPa)")
            return "  ".join(statuses)

        print(f"\nConstraint diagnostic  (V_f={vf:.3f}, σ_y={sigma_y/1e6:.0f}MPa)")
        print("-" * 100)
        for label, (J, V, Vp, sm) in results3.items():
            print(f"  task3 {label:16s}  {_constraint_status(V, Vp, sm, vf, tol, sigma_y)}")
        for label, (J, V, Vp, sm) in results4.items():
            print(f"  task4 {label:16s}  {_constraint_status(V, Vp, sm, vf, tol, sigma_y)}")
