# Bowsprit Topology Optimization — Code & Lecture Connection Guide

This document maps every major technique in `bowsprit_topopt.py` to the lecture
slides and code examples it comes from, explains the math in plain terms, and
notes where the implementation diverges from the lecture examples and why.

---

## 1. Problem setup

**Geometry:** 2D rectangular cantilever, 3.80 × 0.50 m, plane-stress assumption.  
Left edge fully clamped. Two angled tractions on the top face.

| Load | Magnitude | Angle | Location |
|------|-----------|-------|----------|
| F1 | 2 kN | 65° | top edge, strip centred at 0.20 L, half-width 0.025 L |
| F2 | 3 kN | 55° | top edge from 0.95 L to 1.00 L |

Traction = total force / (strip length × thickness). Direction: upper-left,
`(-cos α, sin α)`.

Code: `BowspritLoadProperties`, `BowspritTopOpt._traction_forces()`

---

## 2. Design variable

ρ ∈ [0, 1] lives on a **DG0** (piecewise-constant) function space — one DOF per
mesh cell. Value 1 = solid, 0 = void. MMA optimizes this directly.

```
self._rho = fa.interpolate(fa.Constant(volume_fraction), self._filter_project.Vd)
```

DG0 is required for the parallel DOF mapping to work correctly (see §9).

---

## 3. Helmholtz PDE filter — Lecture 10

**What it does:** Smooths the raw density ρ into ρ̃, removing checkerboard
patterns and enforcing a minimum feature size of roughly 2r.

**PDE:**
```
−r² ∇²ρ̃ + ρ̃ = ρ     on Ω
∂ρ̃/∂n = 0           on ∂Ω   (natural BC)
```

**Weak form (Lecture 10):**
```
∫ r² ∇ρ̃ · ∇v dx + ∫ ρ̃ v dx = ∫ ρ v dx     ∀v
```

**Lecture slide:** Lecture 10, "The double filter approach" slides.

**Code:** `FilterAndProject.filter()` — solves this in **CG1** (P1) space.

**Why CG1, not DG0?**  
DG0 functions have zero gradient *inside* each cell, so `∇ρ̃ = 0` everywhere
and the diffusion term collapses to zero — the filter becomes an identity.
CG1 allows non-zero gradients across cell boundaries, which is what makes
the Helmholtz diffusion actually work.  
The lecture's `filter_and_project.py` uses DG0 for everything; here a separate
`Vf` (CG1) space is used for filter intermediates, while `Vd` (DG0) is kept
for the design variable and projected output.

```python
# FilterAndProject
Vd  = FunctionSpace(mesh, "DG", 0)   # design variable + rho_bar
Vf  = FunctionSpace(mesh, "P",  1)   # filter trial/test functions
```

---

## 4. Heaviside projection — Lecture 10

**What it does:** Maps the smooth field ρ̃ to a near-binary physical density ρ̄.

**Formula (Lecture 10, eq. 23 in the paper):**
```
ρ̄ = [tanh(β η) + tanh(β (ρ̃ − η))] / [tanh(β η) + tanh(β (1 − η))]
```

- β: sharpness. As β → ∞ the projection approaches a step function at ρ̃ = η.
- η: threshold. ρ̃ > η → solid; ρ̃ < η → void.
- Denominator normalises so ρ̄(0) = 0 and ρ̄(1) = 1 for all β.
- Result is projected back to DG0.

**Code:** `FilterAndProject.project(rho_tilde, beta, eta)`

```python
numerator   = tanh(beta*eta) + tanh(beta*(rho_tilde - eta))
denominator = tanh(beta*eta) + tanh(beta*(1.0 - eta))
rho_bar = fa.project(numerator/denominator, self.Vd)
```

---

## 5. Double-filter robust formulation — Lecture 10

**Why:** A single filter+project gives one design realization. If the physical
geometry deviates from the nominal (erosion or dilation of features), performance
can change drastically. The double-filter approach produces multiple realization
designs with the *same topology* under different threshold values, and the
optimizer is required to satisfy constraints for all of them simultaneously.

**Lecture 10, slides 16–18** ("The double filter approach"):  
The key idea from the paper (Christiansen, Lazarov, Jensen & Sigmund, 2015):
first filter+project with η_min (dilated), then filter+project again with each
realization threshold η_k.

**Pipeline:**
```
ρ  →  filter(r)  →  project(β, η_min)  →  filter(r)  →  project(β, η_k)  →  ρ̄_k
```

η_min = min of all η values = dilated threshold (0.45).  
The first projection at η_min creates a dilated intermediate.  
The second projection with η_k creates the final realization.

| Realization | η_k  | Effect |
|-------------|------|--------|
| Dilated (d) | 0.45 | features grow, more material |
| Nominal (i) | 0.50 | balanced |
| Eroded (e)  | 0.55 | features shrink, less material |

**Code:** `FilterAndProject.double_filter(density, beta, eta_min, eta)`

```python
rho_tilde_1 = self.filter(density, self.radius)
rho_bar_1   = self.project(rho_tilde_1, beta, eta_min)
rho_tilde_2 = self.filter(rho_bar_1, self.second_radius)
return        self.project(rho_tilde_2, beta, eta)
```

Called via `BowspritTopOpt.physical_density(beta, eta_values, eta)`.

---

## 6. SIMP stiffness interpolation — Lecture 9 / 10

**What it does:** Assigns each element a Young's modulus based on its density.
Penalises intermediate densities to push the design towards 0 or 1.

**Formula:**
```
E(ρ̄) = E_min + ρ̄ᵖ · (E₀ − E_min)
```

- p = 3: SIMP penalty. A cell at ρ̄ = 0.5 contributes only 0.5³ = 12.5% of E₀.
- E_min = 10⁻⁶ · E₀: prevents a singular stiffness matrix in void regions.

**Code:** `BowspritTopOpt.stresses_with_simp(displacement, rho_bar)`

```python
E   = E_min + rho_bar**self.penalty * (E0 - E_min)
mu  = E / (2.0*(1.0 + nu))
lam = E*nu / (1.0 - nu**2)     # plane-stress Lamé
sigma = lam*tr(eps)*Identity(d) + 2.0*mu*eps
```

**Plane-stress Lamé:** λ_eff = Eν/(1−ν²), **not** the plane-strain λ = Eν/((1+ν)(1−2ν)).  
The lecture's `force_inverter.py:sigma` uses plane-strain; this is the adapted plane-stress version.

---

## 7. Linear elasticity solve — Lecture 3 / 4 + `force_inverter.py`

**Weak form** (plane-stress, with traction loading):
```
a(u,v) = t · ∫ σ(u) : ε(v) dx
L(v)   = t · [∫ T₁·v ds(F1) + ∫ T₂·v ds(F2)]
```

ε = ½(∇u + ∇uᵀ), t = thickness, T₁/T₂ = surface tractions.  
Boundary condition: u = 0 on the clamped left edge.

**Code:** `BowspritTopOpt.solve_topopt(rho_bar)`

```python
a = thickness*inner(stresses_with_simp(u, rho_bar), strains(v))*dx
L = thickness*(dot(T1,v)*ds(F1) + dot(T2,v)*ds(F2))
fa.solve(a == L, displacement, bc, solver_parameters={"linear_solver":"mumps"})
```

**Why MUMPS?** The lecture's `force_inverter.py` uses `gmres + hypre_amg`. With
SIMP at E_min/E₀ = 10⁻⁶ and many void cells, the stiffness matrix condition
number grows as β increases, causing `gmres` to fail with `DIVERGED_ITS`. MUMPS
is a direct solver unaffected by conditioning.

---

## 8. Compliance objective

**Formula:**
```
J_k = ∫ T · u_k ds
```

This is the external work done by the applied tractions. For linear elasticity it
equals strain energy. Minimising J maximises stiffness (minimum-compliance design).

The combined objective sums over all realizations:
```
J = Σ_k J_k(ρ)
```

**Code:** `BowspritTopOpt.compliance(displacement)`

---

## 9. Volume and pitch constraints

**Volume constraint** (applied to dilated realization):
```
V_d = ∫ ρ̄_d dx / (L · H) ≤ V_f = 0.25
```

**Pitch-weighted volume constraint:**
```
V_pitch = ∫ w(x) ρ̄_d dx / ∫ w(x) dx ≤ V_f
w(x) = 1 + α (x/L)²,   α = 2.5
```

Both use the **dilated** realization: if the most-material design satisfies
the constraint, the nominal and eroded designs automatically do too.

**Code:** `BowspritTopOpt.volume_constraint()`, `BowspritTopOpt.pitch_constraint()`

---

## 10. Stress constraint — Lectures 11 and 12

### 10a. The stress singularity problem — Lecture 11 slides 6–7

In void regions ρ̄ → 0, E → E_min ≈ 0. The strain becomes large, and the
stress σ = E · ε can remain finite or even grow, creating a **stress singularity**.
This makes the stress constraint discontinuous at ρ̄ = 0 — optimizers cannot
handle this directly.

**ε-relaxation** (Lecture 11 slide 7) replaces the local constraint
`σ_vM ≤ σ_y` with a relaxed version that naturally vanishes in void regions:

```
f_σ(ρ̄) = ρ̄ / (ε(1−ρ̄) + ρ̄)        ε = 0.2 fixed
```

For ρ̄ = 1 (solid): f_σ = 1.  
For ρ̄ → 0 (void): f_σ → 0, which relaxes the constraint to 0 ≤ σ_y — satisfied trivially.

**Local stress measure (Lecture 11 slide 7):**
```
σ_m = f_σ(ρ̄) · √(σ²_vM + σ²_min)
```

σ_vM is computed with **base-material E** (not SIMP E), so f_σ carries the full
density interpolation. σ_min = 10⁻⁴ · σ_y prevents √0 in void regions.

**Plane-stress von Mises formula used here:**
```
mean  = tr(σ) / 3 = (σ₁₁ + σ₂₂) / 3
s_2D  = σ − mean · I         (in-plane deviatoric)
s_33  = −mean                 (σ₃₃=0 in plane stress but s₃₃≠0)
σ²_vM = 1.5 · (inner(s_2D,s_2D) + mean²)
```

This gives the correct plane-stress result σ²_vM = σ²₁₁ + σ²₂₂ − σ₁₁σ₂₂ + 3σ²₁₂.
Dropping the `mean²` term would underestimate σ_vM by up to ~15%.

**Code:** `BowspritTopOpt.stress_measure()`, `BowspritTopOpt.stresses_base_material()`

### 10b. Artificial stress concentration and β_cap — Lecture 11 slides 8–9

At high β, the Heaviside projection creates sharp 0/1 transitions at jagged
pixel boundaries. These jagged edges produce **artificial stress concentrations**
that do not exist in the real manufactured geometry.

da Silva et al. showed that capping β at β_lim = 2R/l_e (convolution filter) or
β_lim = 4R_PDE/l_e (PDE filter, since R_PDE = R/√3 maps the convolution radius)
limits this effect. Using β ≤ β_lim/2 with ε = 0.2 gives good agreement with
a real CAD geometry.

For this problem: R = 0.04 m, l_e ≈ H/ny = 0.5/30 ≈ 0.017 m →
β_cap = β_lim/2 = R/l_e ≈ 4.8.

When the optimization beta exceeds β_cap, the **stress rho_bar** is computed
with the capped beta, but the compliance objective continues with the full beta.

**Code:** `StressConstraintSettings.beta_cap`, inside `set_up_functionals()`:
```python
beta_s = min(beta, s.beta_cap) if s.beta_cap is not None else beta
```

### 10c. P-mean global aggregation — Lecture 11 slide 10, Lecture 12

One local constraint per mesh element = tens of thousands of constraints for
MMA to handle — infeasible. Replace with **one global constraint** using the
p-mean approximation of the maximum (Lecture 11 slide 10):

```
σ_c^p = (1/|Ω|) ∫_Ω (σ_m / (α · σ_y))^p dΩ
```

As p → ∞, σ_c^p → max(σ_m/(α·σ_y)). The constraint is:
```
σ_c^p − 1 ≤ 0     ⟺    σ_c ≤ 1    for p > 0
```

Using σ_c^p avoids the p-th root, keeping the pyadjoint tape simple.

**p-continuation:** p is raised each β stage (e.g., 2 → 300). At small p,
σ_c^p averages the field (loose constraint, smooth gradients). At large p,
it approaches the true maximum-stress constraint.

**Code:** `BowspritTopOpt.p_mean_stress_constraint(displacement, rho_bar, p, sigma_y)`

```python
sm   = self.stress_measure(displacement, rho_bar, sigma_y, epsilon_relaxation)
area = geometry.length * geometry.height
return fa.assemble((sm / (alpha*sigma_y))**p_const * dx) / area - 1.0
```

**p wrapped in `fa.Constant`:** FEniCS JIT-compiles UFL forms by structure.
A plain Python int creates a new form structure each time p changes → re-compile
every stage. `fa.Constant(float(p))` keeps the structure fixed; only the value changes.

One stress constraint is added **per realization** (dilated, nominal, eroded),
so `ncon` grows from 2 to 2 + len(eta_values).

---

## 11. Adjoint gradients with pyadjoint — `topopt_elasticity_force_inverter.py`

pyadjoint records every FEniCS operation on a **tape**. A `ReducedFunctional`
can then replay the tape in reverse (adjoint solve) to compute dJ/dρ, dV/dρ, etc.,
at the cost of one extra linear solve per functional — no hand-derived adjoint needed.

**Pattern from `topopt_elasticity_force_inverter.py:setUpFunctionals`:**

1. `tape.clear_tape()` — wipe the recording.
2. Run the forward model (filter → project → elasticity → functional).
3. `tape.copy()` — take a snapshot.
4. `tape.optimize(controls=[m], functionals=[J])` — prune irrelevant ops.
5. `ReducedFunctional(J, m, tape=t)` — wrap for adjoint calls.

Each functional gets its own optimized tape copy so `derivative()` only replays
the relevant computation (e.g., volume gradient does not re-solve the elasticity).

**Code:** `_make_reduced_functional()`, `BowspritTopOpt.set_up_functionals()`

```python
def _make_reduced_functional(functional, control):
    tape = fa.get_working_tape()
    t = tape.copy()
    t.optimize(controls=[control], functionals=[functional])
    return fa.ReducedFunctional(functional, control, tape=t)
```

The MMA callbacks `f(x)` and `g(x)` call `Jhat(rho)` and `Jhat.derivative()`
to get the objective value and gradient at each iteration.

---

## 12. MMA optimizer — Lecture 5 + `mma.py`

The Method of Moving Asymptotes (Svanberg 1987) is used to update ρ.
At each iteration MMA builds a convex separable approximation of the objective
and constraints, then solves the subproblem analytically.

Settings (from `topopt_elasticity_force_inverter.py:setUpOptimizer`):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `xmin/xmax` | 0 / 1 | density bounds |
| `move` | 0.1 | max step per iteration |
| `xtol` | 1e-4 | design change convergence |
| `ftol` | 1e-5 | objective convergence |
| `lmax` | 5–15 | max line search iters |
| `kmax` | 20–30 | max MMA iters per β stage |

**Code:** `_make_mma()`, called in `BowspritTopOpt.optimize()` and
`ParallelBowspritOptimizer.optimize()`.

---

## 13. Beta continuation — Lecture 10 + `topopt_elasticity_force_inverter.py`

The Heaviside projection at large β creates a very non-convex landscape.
Starting directly at high β gets stuck in local minima.

**Strategy:** Start with small β (soft Heaviside ≈ linear map), run MMA to
approximate convergence, then double/increase β and repeat. At each β stage
`set_up_functionals` rebuilds the tapes with the new β.

```
β schedule: {1, 2, 3, 4, 5, 8, 12, 16, 24, 36, 72, 108, 144, 288, 432}
```

At the final β = 432 the Heaviside is extremely sharp and the design is
near-binary.

**Code:** `BowspritTopOpt.optimize()` and `ParallelBowspritOptimizer.optimize()`

```python
for stage_idx, beta in enumerate(beta_schedule):
    p_stress = sched[min(stage_idx, len(sched)-1)]
    self.set_up_functionals(beta, eta_values, p_stress, stress_settings)
    mma.solve(rho_petsc)
```

---

## 14. Parallel MPI — Lecture 8 + `parallel.py`

**Motivation:** Three realizations (dilated, nominal, eroded) each require a
full FEM solve. Running them sequentially wastes time. Lecture 8 shows how to
split `MPI.comm_world` into sub-communicator groups, with each group solving
one realization independently.

**Group assignment** (from `parallel.py`):
```
group = global_rank % Ngroups
```

For 12 processes and 3 groups:
```
World ranks:  0  1  2  3  4  5  6  7  8  9  10  11
Group:        0  1  2  0  1  2  0  1  2  0   1   2
Group 0 (η=0.45): ranks {0,3,6,9}   → dilated solve
Group 1 (η=0.50): ranks {1,4,7,10}  → nominal solve
Group 2 (η=0.55): ranks {2,5,8,11}  → eroded solve
```

All groups solve their FEM simultaneously; MMA then gathers results.

**Gathering scalars** (`sgroup2global`): each group broadcasts its scalar
(J_k, V_k, etc.) to all ranks. Returns [J_0, J_1, ...] sorted by group.

**Gathering gradients** (`vgroup2global`): each group computes dJ_k/dρ on
its group communicator, broadcasts to world ranks indexed by global DOF.
`g(x)` sums: dJ/dρ = Σ_k dJ_k/dρ.  
Volume and pitch gradients come only from group 0 (dilated realization).

**Code:** `ParallelBowspritOptimizer`, `Parallel` class (from `parallel.py`).

---

## 15. DOF mapping fix — own fix to `parallel.py`

**The bug in `parallel.py:create_mapping`:**  
`np.isin(coor_all, coor_group)` returns indices in `coor_all`'s sorted order,
not paired to `coor_group[i]`. When the world-comm mesh and the group-comm mesh
partition cells differently, their local DOF orderings diverge, so the pairing
is scrambled — density values get transferred to wrong cells.

**Fix:** Build the mapping by physical coordinates using a `cKDTree` nearest-
neighbour lookup. Each group-local DOF is matched to the world DOF at the
same coordinate, regardless of storage order.

```python
tree = cKDTree(coor_all)
dists, idxs = tree.query(coor_group, k=1)
assert (dists < 1e-8).all()
par.global2group = list(dof_all[idxs])
```

**Code:** `ParallelBowspritOptimizer._build_dof_mapping()`

---

## 16. Class hierarchy

```
CantileverBeam2dLinear          ← beam_configurator_2d.py
│  mesh, function spaces, strains(), stresses(), solve()
│
└── BowspritTopOpt              ← bowsprit_topopt.py
       bowsprit loads, SIMP, filter+project,
       compliance, volume, pitch, stress functionals,
       MMA callbacks f()/g(), beta continuation

    _GroupBowspritTopOpt        ← BowspritTopOpt with pre-made group mesh
    (used internally by ParallelBowspritOptimizer)

ParallelBowspritOptimizer       ← bowsprit_topopt.py
       one _GroupBowspritTopOpt per eta value,
       MPI communicator split, DOF mapping,
       gradient gather, parallel beta continuation
```

---

## 17. Lecture-to-code quick reference

| Lecture | Topic | Code location |
|---------|-------|---------------|
| Lecture 3/4 | Linear elasticity weak form, strain/stress | `BowspritTopOpt.solve_topopt()`, `CantileverBeam2dLinear.strains()` |
| Lecture 5 | MMA optimizer | `_make_mma()`, `mma.py` |
| Lecture 8 | Communicator-split parallel FEM | `ParallelBowspritOptimizer`, `Parallel` class |
| Lecture 9 | SIMP penalization | `BowspritTopOpt.stresses_with_simp()` |
| Lecture 10 | Helmholtz PDE filter | `FilterAndProject.filter()` |
| Lecture 10 | Heaviside projection (eq. 23) | `FilterAndProject.project()` |
| Lecture 10 | Double-filter robust formulation | `FilterAndProject.double_filter()` |
| Lecture 10 | Beta continuation | `BowspritTopOpt.optimize()` |
| Lecture 11 slide 7 | ε-relaxation, f_σ, σ_m formula | `BowspritTopOpt.stress_measure()` |
| Lecture 11 slide 7 | Base-material E for stress | `BowspritTopOpt.stresses_base_material()` |
| Lecture 11 slide 9 | β_cap for artificial concentrations | `StressConstraintSettings.beta_cap` |
| Lecture 11 slide 10 | P-mean global aggregation | `BowspritTopOpt.p_mean_stress_constraint()` |
| Lecture 12 | P-continuation schedule | `StressConstraintSettings.p_stress_schedule` |
| `force_inverter.py` | FEM setup: mesh, SubDomain, Measure, BC | `BowspritTopOpt._mark_end_faces()`, `solve_topopt()` |
| `topopt_elasticity_force_inverter.py` | Tape clear/copy/optimize, ReducedFunctional | `_make_reduced_functional()`, `set_up_functionals()` |
| `parallel.py` | sgroup2global, vgroup2global, fglobal2group | `ParallelBowspritOptimizer.f()`, `g()`, `_sync_rho()` |

---

## 18. Key deviations from lecture code

| Deviation | Reason |
|-----------|--------|
| CG1 filter space instead of DG0 | DG0 kills Helmholtz diffusion term (gradients zero inside cells) |
| MUMPS elasticity solver instead of gmres | SIMP E_min/E₀=1e-6 ill-conditions the system; gmres diverges at high β |
| cKDTree DOF mapping instead of np.isin | np.isin returns matches in coor_all order, scrambling transfer when communicators partition differently |
| Plane-stress Lamé λ = Eν/(1−ν²) | Lecture uses plane-strain λ = Eν/((1+ν)(1−2ν)); bowsprit is thin-walled |
| Per-realization stress constraints | One constraint per η (not one global); ncon = 2 + len(eta_values) |
| β_cap on stress rho_bar only | Compliance uses full β for binary design; stress uses capped β to avoid artificial concentrations |

---

## 19. Running

```bash
# From Exam_handin/
mpiexec -n 12 --use-hwthread-cpus python bowsprit_topopt.py
```

12 = LCM(3, 2) × 4 processes. Task 3 uses 3 groups × 4 cores; Task 4 uses
2 groups × 6 cores. Both divide evenly with no idle processes.

Quick test:
```python
settings = OptimizationSettings(mesh_size=(20,10), beta_schedule=(1,2), kmax=5)
```
