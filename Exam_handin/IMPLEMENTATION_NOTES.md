# Bowsprit Topology Optimization - Implementation Notes

## Problem Statement

The implemented problem is a simplified 2D plane-strain bowsprit spar, modelled
as a rectangular cantilever.

- Domain: `L = 1.0 m`, `H = 0.5 m`
- F1: `2 kN` at `65 deg`, applied on a top-edge strip from `0.15L` to `0.25L`
- F2: `3 kN` at `55 deg`, applied on the top-edge strip from `0.95L` to `1.00L`
- Both tractions point upper-left: `(-cos(alpha), sin(alpha))`
- Material: isotropic CFRP approximation, `E = 70 GPa`, `nu = 0.30`
- Thickness: `t = 0.05 m`
- Volume fraction limit: `0.33`

The solver is written for exam tasks 2-5:

- Task 2: FEniCS + pyadjoint + MMA topology optimization with robust double filtering
- Task 3: three realizations, `eta = {0.4, 0.5, 0.6}`
- Task 4: two realizations, `eta = {0.4, 0.6}`, with `eta = 0.5` evaluated afterwards
- Task 5: compliance and volume comparison of the two approaches

## Code Style And Structure

The current `bowsprit_topopt.py` was refactored to follow the style of
`beam_configurator_2d.py` more closely:

- Uses `import fenics as fs` instead of wildcard imports
- Uses dataclasses for problem data
- Keeps helper methods small and named by their role
- Reuses `CantileverBeam2dLinear` for mesh/function-space conventions, strain calculation, boundary storage, and post-processing state
- Keeps the course-style topology optimization flow: `forward`, `set_up_functionals`, `f`, `g`, `optimize`

The code still uses `fenics_adjoint as fa` for objects that must be visible to
pyadjoint, such as annotated constants, functions, solves, projections, controls,
and reduced functionals.

## Main Classes

### `BowspritLoadProperties`

Stores the load magnitudes, angles, and F1 strip placement.

### `FilterAndProject`

Implements the double-filter method from lecture 10:

```text
rho -> filter(r1) -> project(beta, eta_min)
    -> filter(r2) -> project(beta, eta_k) -> rho_bar_k
```

Current implementation:

- `r1 = filter_radius = 0.04 m`
- `r2 = r1`
- `eta_min = min(eta_values)`
- Same `beta` is used for both projections
- Helmholtz filter:

```text
-r^2 Laplace(rho_tilde) + rho_tilde = rho
```

- Heaviside projection:

```text
rho_bar = [tanh(beta*eta) + tanh(beta*(rho_tilde - eta))]
          / [tanh(beta*eta) + tanh(beta*(1 - eta))]
```

### `BowspritTopOpt`

Main topology optimization class. Important methods:

| Method | Purpose |
| --- | --- |
| `mesh` | Creates a pyadjoint-compatible `RectangleMesh.create(...)` mesh |
| `_mark_end_faces()` | Marks clamp, F1 top strip, and F2 top strip |
| `_traction_forces()` | Converts total forces to boundary tractions |
| `stresses_with_simp()` | Linear elastic stress using SIMP interpolation |
| `physical_density()` | Creates one robust projected density realization |
| `forward()` | Runs density projection and elasticity solve |
| `solve_topopt()` | Solves the SIMP elasticity problem |
| `compliance()` | Computes external work compliance |
| `volume_constraint()` | Computes `V - volume_fraction <= 0` |
| `pitch_constraint()` | Computes right-side mass penalty constraint |
| `set_up_functionals()` | Builds pyadjoint reduced functionals |
| `f()` / `g()` | MMA objective/constraint values and gradients |
| `set_up_history()` | Creates PVD files for ParaView optimization-history animation |
| `plot_k()` | MMA callback that writes history fields during optimization |
| `optimize()` | Runs beta continuation with MMA |
| `evaluate_design()` | Post-processes one realization without tape annotation |
| `save_design()` | Saves `rho_bar` and displacement PVD files |

### `_GroupBowspritTopOpt`

Thin dataclass subclass of `BowspritTopOpt`. Accepts an optional `group_mesh`
argument; if provided, sets `self._mesh` before `__post_init__` creates any
function spaces, so the entire FEM problem lives on the group communicator mesh.

### `ParallelBowspritOptimizer`

Wrapper that follows the eight-step `Parallel` class pattern from Lecture 8.
Splits `MPI.comm_world` into one group per eta realization and runs all FEM
solves concurrently.

| Method | Purpose |
| --- | --- |
| `__post_init__()` | Creates `Parallel`, writes mesh to XDMF, builds world and group beams, sets up DOF mapping |
| `set_up_functionals()` | Builds per-group pyadjoint tapes for J, V, P |
| `_sync_rho()` | Transfers MMA PETSc vector → global rho → group rho (steps 7–8) |
| `f()` / `g()` | Evaluates all groups in parallel, gathers results with `sgroup2global` / `vgroup2global` |
| `optimize()` | Beta continuation with MMA on the world comm |
| `save_design()` | Delegates to the world-comm beam for final evaluation |

## Optimization Formulation

### Task 3: Three Realizations

```text
min_rho    J_d(rho) + J_i(rho) + J_e(rho)
s.t.       V(rho_bar_d) <= 0.33
           V_pitch(rho_bar_d) <= 0.33
           0 <= rho <= 1
```

Both constraints are applied to the dilated realization, since this has the most
material.

### Task 4: Two Realizations

```text
min_rho    J_d(rho) + J_e(rho)
s.t.       V(rho_bar_d) <= 0.33
           V_pitch(rho_bar_d) <= 0.33
           0 <= rho <= 1
```

After optimization, the intermediate design is generated from the final design
variables using `eta = 0.5` and the final `beta`.

## Volume Constraint

The implementation enforces the standard unweighted volume fraction:

```text
V = integral(rho_bar dx) / (L*H)
```

It also enforces a pitch-weighted volume constraint:

```text
V_pitch = integral(w(x)*rho_bar dx) / integral(w(x) dx)
w(x) = 1 + alpha*(x/L)^2
alpha = 2
```

This keeps the normal 33% material constraint, while making material near the tip
more expensive because pitching inertia scales with distance from the support.

## Numerical Choices

| Parameter | Value |
| --- | --- |
| SIMP penalty | `3.0` |
| Ersatz stiffness | `E_min = 1e-6 * E0` |
| Filter radius | `0.04 m` |
| Volume fraction | `0.33` |
| Pitch weight alpha | `2.0` |
| Beta schedule | `[1, 2, 4, 8, 16, 32]` |
| Delta eta | `0.10` |
| Mesh | `(100, 50)` |
| Elasticity solver | `gmres` with `hypre_amg` |
| Filter solver | `cg` with `hypre_amg` |
| Design variable space | `DG0` (one DOF per cell) |

`gmres + hypre_amg` scales across MPI processes. The previous `mumps` default was
a parallel direct solver but its scaling degrades beyond ~8 ranks. `DG0` is
required by `parallel.py`'s DOF-mapping method which is only tested with DG spaces.

## Gradient Computation

For each beta stage, `ParallelBowspritOptimizer.set_up_functionals(beta)` runs
on every MPI process. Each process belongs to one group; each group records a
different eta realization on its own pyadjoint tape:

1. Clears the working tape.
2. Runs one forward solve for this group's eta realization.
3. Copies and optimizes the tape; builds a `ReducedFunctional` for `J_k`.
4. Repeats steps 1–3 for the dilated volume constraint `V`.
5. Repeats steps 1–3 for the pitch-weighted constraint `P`.

All three FEM solves (dilated / nominal / eroded) execute simultaneously across
process groups.

`f(x)` gathers scalars via `sgroup2global`:

```text
J_total = sum(J_0, J_1, J_2)      # sum across all groups
V       = V_0                      # dilated group (group 0)
P       = P_0
```

`g(x)` gathers gradient vectors via `vgroup2global`:

```text
dJ/drho = sum(dJ_0, dJ_1, dJ_2)   # sum across all groups
dV/drho = dV_0
dP/drho = dP_0
```

## Running

```bash
cd Exam_handin
mpiexec -n 12 --use-hwthread-cpus python bowsprit_topopt.py
```

`--use-hwthread-cpus` tells OpenMPI to count hardware threads (16) instead of
physical cores (8). Without it, `mpiexec -n 12` fails with a "not enough slots"
error on the 4800H.

Must run from `Exam_handin/` so local imports (`parallel`, `mma`,
`beam_configurator_2d`) resolve correctly.

`N = 12` is the recommended process count on the 4800H (8 cores / 16 threads):

| Task | Groups | Cores per group |
| --- | --- | --- |
| Task 3 (3 eta values) | 3 | 4 |
| Task 4 (2 eta values) | 2 | 6 |

12 = LCM(3, 2) × 4, so both tasks divide evenly with no idle processes.
To use all 16 threads, run each task separately:

```bash
mpiexec -n 15 --use-hwthread-cpus python bowsprit_topopt.py   # task 3: 3 x 5
mpiexec -n 16 --use-hwthread-cpus python bowsprit_topopt.py   # task 4: 2 x 8
```

Outputs:

- `plots/run_YYYYMMDD_HHMMSS/task3/task3_rho_history.pvd`
- `plots/run_YYYYMMDD_HHMMSS/task3/task3_rho_bar_history.pvd`
- `plots/run_YYYYMMDD_HHMMSS/task3/task3_dilated_rho_bar.pvd`
- `plots/run_YYYYMMDD_HHMMSS/task3/task3_nominal_rho_bar.pvd`
- `plots/run_YYYYMMDD_HHMMSS/task3/task3_eroded_rho_bar.pvd`
- `plots/run_YYYYMMDD_HHMMSS/task4/task4_rho_history.pvd`
- `plots/run_YYYYMMDD_HHMMSS/task4/task4_rho_bar_history.pvd`
- `plots/run_YYYYMMDD_HHMMSS/task4/task4_dilated_rho_bar.pvd`
- `plots/run_YYYYMMDD_HHMMSS/task4/task4_eroded_rho_bar.pvd`
- `plots/run_YYYYMMDD_HHMMSS/task4/task4_intermediate_rho_bar.pvd`
- Matching displacement files ending in `_u.pvd`

Each run gets a new timestamped folder, so old ParaView files are not
overwritten.

The `*_history.pvd` files contain multiple time steps and can be animated in
ParaView using the time controls. `rho_history` shows the raw design variable,
while `rho_bar_history` shows the projected physical density at `eta = 0.5`.
The default history stride is `10`, so every tenth MMA callback is saved.
The stride can be changed without editing code:

```bash
export BOWSPRIT_HISTORY_STRIDE=20
```

For quick testing, reduce the mesh and schedule in `run_exam_problem()`, for
example:

```python
mesh_size=(20, 10)
beta_schedule=[1, 2]
kmax=5
```

## Verification Status

Current checks performed after the refactor:

| Test | Result |
| --- | --- |
| `python -m py_compile Exam_handin/bowsprit_topopt.py` | Passed |
| Instantiate `BowspritTopOpt` on an `8 x 4` mesh | Passed |
| One-stage MMA smoke test with `eta = {0.4, 0.6}` and `kmax = 1` | Passed |
| Smoke-test history output to PVD files | Passed |
| Combined objective `ReducedFunctional` smoke test | Passed |

The full `(100, 50)` exam run has not been re-run after the refactor because it
is expected to take substantially longer.
