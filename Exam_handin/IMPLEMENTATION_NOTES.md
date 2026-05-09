# Bowsprit Topology Optimization — Implementation Notes

## Problem Statement

2D plane-strain bowsprit spar, modelled as a rectangular cantilever.

| Parameter | Value |
|---|---|
| Length L | 3.80 m |
| Height H | 0.50 m |
| Thickness t | 0.05 m |
| F1 magnitude | 2 kN at 65° |
| F1 location | top edge, strip centered at 0.20 L, half-width 0.025 L |
| F2 magnitude | 3 kN at 55° |
| F2 location | top edge from 0.95 L to 1.00 L |
| Force direction | upper-left: `(-cos α, sin α)` for angle α from horizontal |
| Material | isotropic CFRP approximation |
| E₀ | 70 GPa |
| ν | 0.33 |

The left edge is fully clamped. F1 and F2 are converted from total force to surface traction by dividing by the strip area (`strip length × thickness`).

Tasks implemented:

- **Task 2**: FEniCS + pyadjoint + MMA topology optimization with robust double filtering
- **Task 3**: Three realizations, η ∈ {0.45, 0.50, 0.55} solved in parallel
- **Task 4**: Two realizations, η ∈ {0.45, 0.55}, with η = 0.50 evaluated afterwards
- **Task 5**: Compliance and volume comparison of the two approaches

---

## Mathematical Process

### 1. Design variable

The design variable ρ ∈ [0, 1] is defined element-wise on a DG0 function space (one DOF per mesh cell, constant inside each cell). It represents raw material density before filtering.

### 2. Helmholtz filter

The filter converts the raw density into a smooth field ρ̃ by solving the PDE

```
-r² ∇²ρ̃ + ρ̃ = ρ     on Ω
∂ρ̃/∂n = 0           on ∂Ω
```

where r is the filter radius. The weak form is

```
∫ r² ∇ρ̃ · ∇v dx + ∫ ρ̃ v dx = ∫ ρ v dx   ∀v
```

This is solved in a CG1 (P1, piecewise-linear) function space. Using DG0 here would make ∇ρ̃ = 0 inside each element, collapsing the diffusion term to zero and making the filter an identity. CG1 allows non-zero gradients across element boundaries.

The filter has the effect of spreading local density variations over a neighbourhood of radius r, removing checkerboard patterns and imposing a minimum feature size of approximately 2r.

### 3. Heaviside projection

The filtered field ρ̃ is projected to a physical density ρ̄ via the smooth Heaviside

```
ρ̄ = [tanh(β η) + tanh(β (ρ̃ − η))] / [tanh(β η) + tanh(β (1 − η))]
```

- β controls the sharpness of the transition. As β → ∞, the projection approaches a step function at ρ̃ = η.
- η is the threshold. Values of ρ̃ above η map to ρ̄ → 1 (solid); below η map to ρ̄ → 0 (void).
- The denominator normalises so that ρ̄(ρ̃=0) = 0 and ρ̄(ρ̃=1) = 1 hold at all β.

The result ρ̄ is projected back to the DG0 space.

### 4. Double-filter (robust formulation)

A single filter+project gives the **nominal** design. The robust formulation applies filter and projection twice:

```
ρ  →  filter(r₁)  →  project(β, η_min)  →  filter(r₂)  →  project(β, η_k)  →  ρ̄_k
```

- First filter+project (with η_min = min of all η values) creates a preliminary smoothed design.
- Second filter+project with the realization-specific η_k creates the final physical density ρ̄_k.

For Task 3, three realizations are produced:

| Realization | η_k | Effect |
|---|---|---|
| Dilated (d) | 0.45 | Features grow wider → more material |
| Nominal (i) | 0.50 | Balanced intermediate |
| Eroded (e) | 0.55 | Features shrink → less material |

For Task 4, only dilated (η=0.45) and eroded (η=0.55) are optimized. The nominal (η=0.50) is computed afterwards from the final design variables without re-optimization.

In both cases r₁ = r₂ = filter_radius = 0.04 m.

### 5. SIMP stiffness interpolation

The elasticity modulus of each element is interpolated using SIMP (Solid Isotropic Material with Penalization):

```
E(ρ̄) = E_min + ρ̄^p · (E₀ − E_min)
```

- p = 3 is the penalty exponent. It penalizes intermediate densities: a cell with ρ̄ = 0.5 contributes only 0.5³ = 12.5% of E₀ to stiffness, discouraging grey material.
- E_min = 10⁻⁶ · E₀ prevents a singular stiffness matrix in fully void regions.

### 6. Linear elasticity solve

For each realization k, the displacement u_k is found by solving the plane-strain linear elasticity problem

```
∫ σ(u_k) : ε(v) dx = ∫ T · v ds     ∀v ∈ V
```

where ε = ½(∇u + ∇uᵀ) is the small-strain tensor, σ = λ tr(ε) I + 2μ ε is the Cauchy stress (with Lamé constants λ = E ν / ((1+ν)(1−2ν)), μ = E / (2(1+ν))), and T is the boundary traction from F1 and F2.

Boundary conditions: u = 0 on the clamped left edge.

Solver: MUMPS direct solver (reliable for ill-conditioned SIMP systems where E_min/E₀ = 10⁻⁶ causes condition numbers that defeat iterative solvers at high β).

### 7. Compliance objective

The structural compliance (external work) is

```
J_k(ρ) = ∫ T · u_k ds
```

This is the work done by the applied tractions. Minimizing compliance maximizes stiffness.

The combined objective across realizations is the sum:

```
J = Σ_k J_k(ρ)
```

### 8. Volume constraint

Standard unweighted volume fraction (applied to dilated realization):

```
V = ∫ ρ̄_d dx / (L · H) ≤ Vf
```

where Vf = 0.25.

### 9. Pitch constraint

A pitch-weighted volume constraint penalizes material far from the support, because pitching inertia scales with the square of distance from the clamped root:

```
V_pitch = ∫ w(x) ρ̄_d dx / ∫ w(x) dx ≤ Vf
w(x) = 1 + α (x/L)²,    α = 2.5
```

Both constraints use the dilated realization (most material), so satisfying them for the dilated design guarantees they hold for the nominal and eroded designs too.

### 10. Adjoint gradient computation

Gradients of J, V, P with respect to ρ are computed via pyadjoint. Each forward solve is recorded on a tape. A `ReducedFunctional` object then differentiates the tape in reverse (adjoint solve) to give dJ/dρ, dV/dρ, dP/dρ at the cost of one additional linear solve per functional. No hand-derived adjoint is needed.

### 11. MMA optimization

The Method of Moving Asymptotes (MMA, Svanberg 1987) is used to update ρ. MMA is a gradient-based optimizer that constructs a convex separable approximation of the objective and constraints at each iteration, then solves the subproblem analytically.

Settings: move limit 0.10, max 40 iterations per β stage.

### 12. Beta continuation

β is ramped through the schedule {1, 2, 4, 8, 16, 32, 64, 128, 256}. At each β, MMA is run to approximate convergence before β is doubled. This continuation strategy prevents the optimizer from getting stuck in a local minimum caused by the sharp Heaviside at large β, while still converging to a near-binary design at β = 256.

---

## Source Map: Lecture File Origins

| Component | Source file |
|---|---|
| `Parallel` class (MPI group splitting, DOF mapping, `sgroup2global`, `vgroup2global`) | `Lecture slides/.../parallel.py` (verbatim copy) |
| `MMA_petsc` optimizer | `mma.py` (by Søren Madsen, course material) |
| `FilterAndProject` interface (filter, project, double-filter API) | `Lecture slides/.../filter_and_project.py` |
| `FilterAndProject` math (Helmholtz PDE + Heaviside formula) | Lecture 10 slides |
| FEM setup: mesh, `VectorFunctionSpace`, `MeshFunction`, `SubDomain`, `Measure`, `DirichletBC` | `Lecture slides/.../force_inverter.py` |
| Elasticity weak form, `sigma`, `epsilon`, `solve` calls | `Lecture slides/.../force_inverter.py` |
| Optimization flow: `setUpFunctionals`, tape clear/copy/optimize, `ReducedFunctional`, `f(x)`, `g(x)`, beta loop | `Lecture slides/.../topopt_elasticity_force_inverter.py` |
| `ParallelBowspritOptimizer` eight-step pattern | `Lecture slides/.../topopt_elasticity_force_inverter_parallel.py` + Lecture 8 slides |
| `fixParts` / fixed-density regions | `Lecture slides/.../topopt_elasticity_force_inverter.py` |
| Dataclass style, `CantileverBeam2dLinear` base class | `beam_configurator_2d.py` (earlier course assignment) |
| Problem geometry, loads, and task formulation | `Exam2026.pdf` |

### Key adaptations from lecture code

**`parallel.py` DOF mapping bug fix**: `parallel.py`'s `create_mapping` uses `np.isin(coor_all, coor_group)` which returns indices in `coor_all`'s sorted order rather than paired to `coor_group[i]`. When a world-comm mesh and a group-comm mesh partition cells differently, the local DOF orderings diverge, producing a scrambled density transfer. `_build_dof_mapping` replaces this with a `scipy.spatial.cKDTree` nearest-neighbour lookup, matching each group coordinate to its world counterpart by distance.

**CG1 filter space**: The lecture's `filter_and_project.py` uses DG0 for both the design variable and filter intermediates. DG0 functions have zero gradient inside each element, so the diffusion term `r²⟨∇ρ̃,∇v⟩·dx = 0` and the filter reduces to an identity. `FilterAndProject` here uses a separate CG1 (`Vf`) space for filter trial/test functions and output, while keeping DG0 (`Vd`) for the design variable and projected result.

**MUMPS elasticity solver**: The lecture's `force_inverter.py` uses gmres + hypre_amg. For SIMP with E_min/E₀ = 10⁻⁶ and many void cells, the stiffness matrix becomes severely ill-conditioned at large β, causing gmres to diverge with `DIVERGED_ITS`. Switching to MUMPS (a direct solver) eliminates this.

---

## Numerical Parameters

| Parameter | Value |
|---|---|
| SIMP penalty p | 3 |
| E_min | 1e-6 · E₀ |
| Filter radius r | 0.04 m |
| Volume fraction limit Vf | 0.25 |
| Pitch weight α | 2.5 |
| η_d (dilated) | 0.45 |
| η_i (nominal) | 0.50 |
| η_e (eroded) | 0.55 |
| Beta schedule | {1, 2, 4, 8, 16, 32, 64, 128, 256} |
| MMA move | 0.10 |
| MMA kmax per stage | 40 |
| Mesh (default) | (230, 30) cells |
| Mesh (exam run) | (460, 60) cells → 55 200 triangles |
| Elasticity solver | MUMPS (direct) |
| Filter solver | CG + hypre_amg |
| Design variable space | DG0 (required by `Parallel` DOF mapping) |
| Filter intermediate space | CG1 (required for non-zero Helmholtz diffusion) |

---

## Code Structure

### `BowspritLoadProperties`

Stores total force magnitudes, angles, and strip placement fractions for F1 and F2.

### `FilterAndProject`

Implements the double-filter method.

- `Vd`: DG0 function space for design variable and projected density
- `Vf`: CG1 function space for Helmholtz filter intermediates
- `filter(density, radius)`: solves Helmholtz PDE in CG1
- `project(rho_tilde, beta, eta)`: smooth Heaviside, projects result to DG0
- `double_filter(density, beta, eta_min, eta)`: applies filter→project→filter→project

### `BowspritTopOpt`

Single-group SIMP solver. Key methods:

| Method | Purpose |
|---|---|
| `mesh` | Creates `RectangleMesh.create` (pyadjoint-compatible) |
| `_mark_end_faces` | Marks CLAMP (1), F1 strip (2), F2 strip (3) |
| `_traction_forces` | Converts total force to `Constant((Tx, Ty))` traction |
| `stresses_with_simp` | SIMP stiffness E(ρ̄) + plane-strain σ |
| `physical_density` | Calls `double_filter` for one realization |
| `forward` | Runs physical density + elasticity solve |
| `compliance` | External work `∫ T · u ds` |
| `volume_constraint` | `∫ ρ̄ dx / (L·H) − Vf` |
| `pitch_constraint` | Pitch-weighted volume `∫ w ρ̄ dx / ∫ w dx − Vf` |
| `set_up_functionals` | Builds pyadjoint tapes + `ReducedFunctional` for J, V, P |
| `f` / `g` | MMA objective/constraint values and gradients |
| `set_up_history` | Creates PVD files for ParaView animation |
| `plot_k` | MMA callback, writes density history every N iterations |
| `evaluate_design` | Post-processes one realization without tape annotation |
| `save_design` | Saves ρ̄ and u PVD files |

### `ParallelBowspritOptimizer`

Wraps multiple `BowspritTopOpt` instances on MPI communicator-split groups. One group per η realization; all groups run their FEM solves simultaneously.

| Method | Purpose |
|---|---|
| `__post_init__` | Creates `Parallel`, builds world and group meshes, sets up DOF mapping |
| `_build_dof_mapping` | cKDTree coordinate lookup replacing `parallel.py`'s `np.isin` |
| `set_up_functionals` | Builds per-group pyadjoint tapes for J, V, P |
| `_sync_rho` | Transfers MMA PETSc vector → global ρ → group ρ |
| `f` / `g` | Evaluates all groups in parallel, gathers with `sgroup2global` / `vgroup2global` |
| `optimize` | Beta continuation with MMA on world comm |
| `save_design` | Delegates final design evaluation to world-comm beam |

---

## Parallel Execution

`Parallel` splits `MPI.comm_world` into N groups by assigning rank r to group `r % N`. Each group runs its FEM solve simultaneously:

```
World ranks:  0  1  2  3  4  5  6  7  8  9  10  11
Task 3 (N=3): 0  1  2  0  1  2  0  1  2  0   1   2   ← group index
Group 0 (η=0.45): ranks {0,3,6, 9}  → dilated solve
Group 1 (η=0.50): ranks {1,4,7,10}  → nominal solve
Group 2 (η=0.55): ranks {2,5,8,11}  → eroded solve
```

**Gradient gathering** (`vgroup2global`):

1. Each group computes dJ_k/dρ on its group communicator.
2. `vgroup2global` gathers all group gradients to all world ranks, indexed by global DOF.
3. `g(x)` sums across groups: `dJ/dρ = Σ_k dJ_k/dρ`.
4. Volume and pitch constraint gradients come only from group 0 (dilated realization).

**Scalar gathering** (`sgroup2global`):

Each group computes its scalar (J_k, V_k, or P_k) and `sgroup2global` returns the vector [J_0, J_1, ...] to all ranks. The master assembles `J = Σ J_k`, `V = V_0`, `P = P_0`.

---

## Running

```bash
cd Exam_handin
mpiexec -n 12 --use-hwthread-cpus python bowsprit_topopt.py
```

`--use-hwthread-cpus` tells OpenMPI to count hardware threads (16 on 4800H) instead of physical cores (8). Without it, `mpiexec -n 12` may fail with "not enough slots".

Recommended process counts on Ryzen 7 4800H (8 cores / 16 threads):

| Task | Groups | Processes | Cores per group |
|---|---|---|---|
| Task 3 (3 η values) | 3 | 12 | 4 |
| Task 4 (2 η values) | 2 | 12 | 6 |

12 = LCM(3, 2) × 4, so both tasks divide evenly with no idle processes.

For maximum threads per task (run separately):

```bash
mpiexec -n 15 --use-hwthread-cpus python bowsprit_topopt.py   # task 3: 3 × 5
mpiexec -n 16 --use-hwthread-cpus python bowsprit_topopt.py   # task 4: 2 × 8
```

Control history stride (default 10 iterations between saves):

```bash
export BOWSPRIT_HISTORY_STRIDE=20
```

Quick test with smaller mesh:

```python
settings = OptimizationSettings(mesh_size=(20, 10), beta_schedule=(1, 2), kmax=5)
```

---

## Output Files

All output goes under `plots/run_YYYYMMDD_HHMMSS/`. Each run gets a timestamped folder so old ParaView files are not overwritten.

```
plots/run_YYYYMMDD_HHMMSS/
├── boundary_parts.pvd            ← marked face regions
├── task3/
│   ├── task3_rho_history.pvd     ← raw ρ animation (every N MMA iters)
│   ├── task3_rho_bar_history.pvd ← projected ρ̄ animation at η=0.50
│   ├── task3_dilated_rho_bar.pvd ← final dilated design
│   ├── task3_nominal_rho_bar.pvd ← final nominal design
│   ├── task3_eroded_rho_bar.pvd  ← final eroded design
│   └── task3_*_u.pvd             ← matching displacement files
└── task4/
    ├── task4_rho_history.pvd
    ├── task4_rho_bar_history.pvd
    ├── task4_dilated_rho_bar.pvd
    ├── task4_eroded_rho_bar.pvd
    ├── task4_intermediate_rho_bar.pvd ← η=0.50 evaluated post-optimization
    └── task4_*_u.pvd
```

The `*_history.pvd` files contain multiple time steps and can be animated in ParaView using the time controls.

---

## Verification Status

| Test | Result |
|---|---|
| `python -m py_compile Exam_handin/bowsprit_topopt.py` | Passed |
| Instantiate `BowspritTopOpt` on 8×4 mesh | Passed |
| One-stage MMA smoke test with η ∈ {0.45, 0.55}, kmax=1 | Passed |
| Smoke-test history output to PVD files | Passed |
| `ParallelBowspritOptimizer` with 12 ranks, beta through 8 | Passed (gmres fixed → mumps; DOF mapping fixed → cKDTree; filter fixed → CG1) |
| Full (460, 60) exam run with beta to 256 | In progress |
