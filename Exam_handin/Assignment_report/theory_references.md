# Theory References for `bowsprit_topopt.py` and `parallel.py`

Each section maps a code component to its lecture source, states the relevant mathematics, and justifies the chosen numerical values.

---

## 1. Linear Elasticity BVP — `solve_topopt`

**Source: Lecture 3, p. 7–9**

### Strong form

The governing PDE for a linear elastic body occupying domain Ω is:

$$-\nabla \cdot \boldsymbol{\sigma} = \mathbf{f} \quad \text{in } \Omega$$

with constitutive law (Lecture 3, p. 7):

$$\boldsymbol{\sigma} = \lambda \, \mathrm{tr}(\boldsymbol{\varepsilon})\,\mathbf{I} + 2\mu\,\boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} = \tfrac{1}{2}\!\left(\nabla\mathbf{u} + (\nabla\mathbf{u})^\top\right)$$

Boundary conditions: Dirichlet $\mathbf{u}=\mathbf{0}$ on $\Gamma_D$ (clamped root), Neumann traction $\boldsymbol{\sigma}\cdot\mathbf{n}=\mathbf{T}$ on $\Gamma_N$ (applied load).

### Weak form (Lecture 3, p. 8–9)

Multiply by test function $\mathbf{v} \in V$ and integrate by parts. Traction BCs enter as natural boundary terms:

$$\int_\Omega \boldsymbol{\sigma}(\mathbf{u}) : \boldsymbol{\varepsilon}(\mathbf{v})\, d\Omega = \int_\Omega \mathbf{f}\cdot\mathbf{v}\, d\Omega + \int_{\Gamma_N} \mathbf{T}\cdot\mathbf{v}\, d\Gamma \qquad \forall\,\mathbf{v} \in V$$

In standard notation: $a(\mathbf{u},\mathbf{v}) = L(\mathbf{v})$ where $a$ is the bilinear stiffness form and $L$ is the load linear form.

### Plane stress (Lecture 3, p. 7, item 6c)

The bowsprit spar is thin out-of-plane ($t = 0.05\,\text{m}$) relative to its span ($L = 3.80\,\text{m}$). The correct assumption is **plane stress** ($\sigma_{33} = 0$, $\varepsilon_{33} \neq 0$), which gives effective Lamé parameters:

$$\lambda_\text{eff} = \frac{E\nu}{1-\nu^2}, \qquad \mu = \frac{E}{2(1+\nu)}$$

**Plane strain** (incorrect for this geometry) would use $\lambda = E\nu/((1+\nu)(1-2\nu))$.

### Compliance objective

In linear elasticity, external work equals strain energy: $J = \int_{\Gamma_N}\mathbf{T}\cdot\mathbf{u}\,d\Gamma = \int_\Omega \boldsymbol{\sigma}:\boldsymbol{\varepsilon}\,d\Omega$. Minimising compliance $J$ is equivalent to maximising global stiffness.

### MUMPS direct solver

SIMP (§2) creates a condition number ratio $E_{min}/E_0 = 10^{-6}$ between void and solid elements. Iterative solvers (CG, GMRES) struggle with such ill-conditioned systems, especially at high $\beta$ where the density field is nearly binary. MUMPS (Multifrontal Massively Parallel Sparse direct Solver) is used instead, as shown in Lecture 3, p. 23 example code.

---

## 2. SIMP Density Interpolation — `stresses_with_simp`, `_simp_E`

**Source: Lecture 6, p. 9; Lecture 8, p. 4**

### Modified SIMP formula

$$E(\bar{\rho}) = E_\min + \bar{\rho}^p (E_0 - E_\min)$$

This is the **modified SIMP** scheme (Lecture 6, p. 9). The modification $+E_\min$ prevents the stiffness matrix $\mathbf{K}$ from becoming rank-deficient when elements have $\bar{\rho} = 0$.

### Penalty $p = 3$

Lecture 6, p. 9 states: *"p = 3 or higher works well in practice; in practice $3 \leq p \leq 7$."*

At intermediate density $\bar{\rho} = 0.5$:
$$E(0.5) = E_\min + 0.5^3(E_0 - E_\min) \approx 0.125\,E_0$$

The isotropic interpolation would give $E = 0.5\,E_0$. Since the SIMP element costs only $0.125\,E_0$ worth of stiffness but uses $0.5\,V_0$ of material, intermediate densities are structurally inefficient. The optimiser is therefore driven toward $\bar{\rho} \in \{0,1\}$, producing near-binary (manufacturable) designs.

### $E_\min = 10^{-6} E_0$

Prevents singular $\mathbf{K}$ in void cells. Value $10^{-6}$ is standard (Lecture 6, p. 9 and force-inverter lecture example): small enough that void elements contribute negligible stiffness, large enough for numerical stability.

> **Note on code comment:** `stresses_with_simp` references "Lecture 9/10". This is incorrect — Lecture 9 covers acoustic wave equations. The correct source is **Lecture 6, p. 9**.

---

## 3. Helmholtz PDE Filter — `FilterAndProject.filter`

**Source: Lecture 6, p. 12; Lecture 8, p. 18**

### PDE and boundary condition

The filter converts raw design field $\rho$ into smoothed field $\tilde{\rho}$ by solving (Lecture 6, p. 12):

$$-r^2 \nabla^2 \tilde{\rho} + \tilde{\rho} = \rho \quad \text{in } \Omega, \qquad \frac{\partial \tilde{\rho}}{\partial n} = 0 \quad \text{on } \partial\Omega$$

The Neumann condition means no material "flux" crosses the boundary — the filter does not introduce artificial source/sink effects at walls.

### Weak form

Multiply by test function $v$ and integrate by parts. The Neumann BC is a natural condition: the boundary term $\oint r^2 \nabla\tilde{\rho}\cdot\mathbf{n}\,v\,dS = 0$ vanishes automatically. The weak form is:

$$\int_\Omega r^2 \nabla\tilde{\rho}\cdot\nabla v\,d\Omega + \int_\Omega \tilde{\rho}\,v\,d\Omega = \int_\Omega \rho\,v\,d\Omega \qquad \forall\,v$$

### Filter radius $r = 0.04\,\text{m}$

The Helmholtz PDE filter is equivalent to a convolution filter with radius $R_\text{conv} = 2\sqrt{3}\,r$ (Lecture 8, p. 18). Here:

$$R_\text{conv} = 2\sqrt{3} \times 0.04 \approx 0.139\,\text{m}$$

Minimum feature size is approximately $2r = 0.08\,\text{m}$ (4.8 cm). For the bowsprit geometry ($L = 3.80\,\text{m}$, $H = 0.50\,\text{m}$) this is 2% of length — a physically sensible structural resolution.

### Why CG1 (not DG0) function space

DG0 (piecewise constant, discontinuous) basis functions have zero gradient *within* each cell. The diffusion term $r^2\nabla\tilde{\rho}\cdot\nabla v$ would vanish everywhere → no filtering. CG1 (piecewise linear, continuous) elements have nonzero inter-cell gradients, so the filter PDE is properly assembled. This is consistent with Lecture 6, p. 12, which uses continuous elements in its FEniCS example.

---

## 4. Heaviside Projection — `FilterAndProject.project`

**Source: Lecture 6, p. 14; Lecture 8, p. 7 eq. (9)**

### Formula

$$\bar{\rho} = \frac{\tanh(\beta\eta) + \tanh\!\left(\beta(\tilde{\rho}-\eta)\right)}{\tanh(\beta\eta) + \tanh\!\left(\beta(1-\eta)\right)}$$

The denominator normalises so that $\bar{\rho} \in [0,1]$ and $\bar{\rho}(\eta) = 0.5$ exactly (Lecture 8, p. 7 eq. 9).

### Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| $\eta$ | threshold | $\bar{\rho}(\tilde{\rho}=\eta) = 0.5$; controls solid/void balance |
| $\beta$ | sharpness | $\beta \to 0$: linear interpolation; $\beta \to \infty$: step function |

At finite $\beta$ the projection is smooth and differentiable — adjoint gradients required by MMA exist everywhere.

---

## 5. Robust Three-Realization Formulation — `double_filter`, `physical_density`

**Source: Lecture 8, p. 9 eq. (14), p. 27; Lecture 10, p. 10, 16**

### Standard robust formulation (Lecture 8, p. 9 eq. 14)

Wang, Lazarov & Sigmund (2011) introduce three realizations of the same design variable $\rho$ by projecting with three different $\eta$ values:

| Realization | $\eta$ | Effect |
|------------|--------|--------|
| Dilated | $\eta_d = \eta$ (small) | more material than nominal |
| Nominal | $\eta_i = 0.5$ | balanced |
| Eroded | $\eta_e = 1-\eta$ (large) | less material than nominal |

The robust problem is (Lecture 8, p. 9 eq. 14):

$$\min_\rho \max\!\left(J(\bar{\rho}^e),\,J(\bar{\rho}^i),\,J(\bar{\rho}^d)\right) \quad \text{s.t.} \quad \frac{1}{|\Omega|}\int_\Omega \bar{\rho}^d\,d\Omega \leq V_f$$

Volume constraint is placed on the **dilated** realization: dilated $\leq V_f$ implies nominal and eroded also $\leq V_f$ because dilation always adds material relative to nominal. (Lecture 8, p. 27 note: *"Volfrac is the volume fraction for the dilated design."*)

### $\eta$ values in code: $\eta_d=0.45$, $\eta_i=0.50$, $\eta_e=0.55$, i.e. $\Delta\eta = 0.05$

Lecture 8, p. 34 implementation example uses $\Delta\eta = 0.2$ (for a force inverter). The bowsprit uses $\Delta\eta = 0.05$ — a tighter manufacturing tolerance reflecting that small deviations from the designed boundary should still give acceptable structural performance.

### Double-filter pipeline (Lecture 10, p. 10, 16)

Standard robust formulation can produce topologically different designs for different $\eta$ values when the filtered field $\tilde{\rho}$ has many intermediate values. The double-filter approach (Christiansen, Lazarov, Jensen & Sigmund, 2015) resolves this by adding a second filter–project pass that pre-binarises the field:

$$\rho \xrightarrow{\text{filter}(r_1)} \tilde{\rho}_1 \xrightarrow{\text{project}(\beta,\,\eta_{min})} \bar{\rho}_\text{pre} \xrightarrow{\text{filter}(r_2)} \tilde{\rho}_2 \xrightarrow{\text{project}(\beta,\,\eta_k)} \bar{\rho}_k$$

The first projection uses $\eta_1 = \min(\eta_\text{values}) = \eta_d = 0.45$ — the dilated threshold — to produce a consistently solid pre-field. The second projection then applies each realization threshold $\eta_k \in \{\eta_d, \eta_i, \eta_e\}$ to create fine-grained erosion/dilation (code: `min(eta_values)` in `double_filter`).

**Deviation from paper recommendation:** Lecture 10, p. 16 recommends $r_2 = \tfrac{1}{2}r_1$ and $\beta_2 = \tfrac{1}{2}\beta_1$. The bowsprit code uses $r_2 = r_1 = 0.04\,\text{m}$ and the same $\beta$ for both passes. This simplification is reasonable for a structural compliance problem which is less topologically sensitive than the acoustic scattering problems the paper targets.

---

## 6. Beta Schedule and Continuation — `BowspritOptimizer`

**Source: Lecture 6, p. 17; Lecture 8, p. 10**

### Schedule

```
β = (1, 2, 3, 4, 5, 8, 12, 16, 24, 36, 72, 108, 144, 288, 432)
```

15 stages, 30 MMA iterations per stage.

### Why start at $\beta = 1$

At $\beta = 1$ the Heaviside projection is nearly linear. The optimiser has full design freedom; intermediate densities are penalised only weakly by SIMP. As $\beta$ increases, cells with $\tilde{\rho}$ far from $\eta$ approach 0 or 1, their sensitivity $\partial\bar{\rho}/\partial\tilde{\rho} \to 0$, and the design is progressively frozen. Starting low prevents early convergence to poor local minima (Lecture 6, p. 17).

### Why $\beta_\max = 432$, not the lecture's "32–128"

Lecture 6, p. 17 suggests $\beta_\max \approx 32$–$128$. The bowsprit uses a much higher value because of the **double filter**: the output field passes through two Heaviside projections in series. The effective sharpness at the output is approximately $\beta$ applied twice, so the raw $\beta$ input must be increased to achieve equivalent binarisation. $\beta = 432$ ensures the final design is nearly perfectly binary.

### Adaptive vs fixed schedule

Lecture 8, p. 10 describes doubling $\beta$ when $\|\Delta\rho\|_\infty \leq 0.01$ or every 50 iterations (adaptive). The bowsprit code uses a fixed schedule for reproducibility.

---

## 7. Beta-Cap for Stress — `StressConstraintSettings.beta_cap`

**Source: Lecture 11, p. 9**

### Problem

At sharp Heaviside interfaces (high $\beta$), the discrete displacement field has steep gradients that the stress computation amplifies. These are numerical artefacts of the mesh, not physical stress concentrations.

### Formula (Lecture 11, p. 9)

For a convolution filter with radius $R$, the maximum permissible $\beta$ before artefacts appear is:

$$\beta_{lim}^\text{conv} = \frac{2R}{l_e}$$

For the Helmholtz PDE filter, using the equivalence $R = 2\sqrt{3}\,r$ (Lecture 8, p. 18) and a safe factor:

$$\beta_{lim}^\text{PDE} = \frac{4r}{l_e}, \qquad \beta_\text{cap} = \frac{\beta_{lim}}{2} = \frac{2r}{l_e}$$

### Code value $\beta_\text{cap} = 4.8$

With $r = 0.04\,\text{m}$ and element size $l_e = H/n_y$:

| Mesh $n_y$ | $l_e$ | $\beta_\text{cap} = 2r/l_e$ |
|-----------|-------|--------------------------|
| 30 | 0.0167 m | **4.8** (default in code) |
| 60 | 0.00833 m | **9.6** (noted in code comment) |

The default `StressConstraintSettings.beta_cap = 4.8` is appropriate for $n_y = 30$; the code comment mentions 9.6 for the finer mesh. The value 4.8 is also more conservative than the formula strictly requires, which eliminates all artefacts.

---

## 8. Stress Constraint Pipeline

**Source: Lecture 11, p. 3–11; Lecture 12, p. 4**

### 8a. Why stress constraints are needed

The point-wise constraint is $\sigma_\text{vM}(x) \leq \sigma_y$ for all $x \in \Omega$. This cannot simply be enforced element-by-element in SIMP because as $\bar{\rho} \to 0$ (void), the displacement field grows large (nearly unconstrained), and the computed stress diverges — the **stress singularity** (Lecture 11, p. 6). The $\varepsilon$-relaxation below removes this singularity.

### 8b. Physical stress uses base modulus $E_0$ (Lecture 11, p. 7)

The stress is computed using the base (solid) stiffness $E_0$, not the SIMP-interpolated $E(\bar{\rho})$:

$$\boldsymbol{\sigma}_\text{phys} = \frac{E_0}{1-\nu^2}\left[\nu\,\mathrm{tr}(\boldsymbol{\varepsilon})\mathbf{I} + (1-\nu)\boldsymbol{\varepsilon}\right] \quad \text{(plane stress)}$$

Using SIMP $E$ here would double-count density: $f_\sigma(\bar{\rho})$ (§8c) already accounts for density interpolation.

### 8c. $\varepsilon$-relaxation — `stress_measure` (Lecture 11, p. 7)

$$f_\sigma(\bar{\rho}) = \frac{\bar{\rho}}{\varepsilon(1-\bar{\rho}) + \bar{\rho}}$$

Behaviour:
- $\bar{\rho} = 0$: $f_\sigma = 0$ — void contributes zero stress regardless of displacement
- $\bar{\rho} = 1$: $f_\sigma = 1$ — full material, full stress
- Smooth monotone function → adjoint gradient well-defined everywhere

The relaxed stress measure is:

$$\sigma_m = f_\sigma(\bar{\rho})\cdot\max\!\left(\sigma_\text{vM},\,\sigma_\text{min}\right)$$

**$\varepsilon = 0.2$:** Lecture 11, p. 9 explicitly recommends $\varepsilon = 0.2$.

**$\sigma_\text{min} = 10^{-4}\sigma_y$:** Prevents division by zero in void elements where $\sigma_\text{vM}$ would be numerically zero. Sets a floor so $\sigma_m \geq 10^{-4}\sigma_y$ always.

### 8d. Von Mises with out-of-plane correction — `stress_measure`

**Source: Lecture 3, p. 10; Lecture 11, p. 7**

In plane stress $\sigma_{33} = 0$, but the deviatoric component $s_{33} \neq 0$:

$$s_{33} = \sigma_{33} - \frac{\sigma_{11}+\sigma_{22}+\sigma_{33}}{3} = -\frac{\sigma_{11}+\sigma_{22}}{3} = -\bar{\sigma}$$

where $\bar{\sigma} = (\sigma_{11}+\sigma_{22})/3$ is the mean stress (2D trace, since $\sigma_{33}=0$).

The full deviatoric tensor contraction is:

$$\mathbf{s}:\mathbf{s} = \underbrace{s_{11}^2 + 2s_{12}^2 + s_{22}^2}_{\text{inner}(\mathbf{s}_\text{2D},\mathbf{s}_\text{2D})} + s_{33}^2 = \text{inner}(\mathbf{s},\mathbf{s}) + \bar{\sigma}^2$$

Von Mises stress squared:

$$\sigma_\text{vM}^2 = \frac{3}{2}\,\mathbf{s}:\mathbf{s} = \frac{3}{2}\!\left(\text{inner}(\mathbf{s},\mathbf{s}) + \bar{\sigma}^2\right)$$

This matches the code: `sigma_vm2 = 1.5*(fs.inner(s,s) + mean_stress**2)`.

> **Note:** The Lecture 3 FEniCS example (elasticity\_traction.py) uses `sqrt(3/2*inner(s,s))`, which neglects $s_{33}$. This is an incomplete formula for plane stress. The bowsprit code includes the $\bar{\sigma}^2$ term, giving the correct full 3D Von Mises value.

### 8e. P-mean global stress constraint — `p_mean_stress_constraint`

**Source: Lecture 11, p. 10–11; Lecture 12, p. 4**

A single global constraint replaces the infinite set of point-wise constraints:

$$\sigma_c^{(p)} = \left(\frac{1}{|\Omega|} \int_\Omega \left(\frac{\sigma_m}{\alpha\,\sigma_y}\right)^p d\Omega\right)^{1/p} \leq 1$$

As $p \to \infty$, $\sigma_c^{(p)} \to \max_x \sigma_m/(\alpha\sigma_y)$. At finite $p$, the p-mean is a smooth differentiable approximation to the maximum, making it compatible with gradient-based MMA.

**$\alpha = 1.0$:** No additional scaling. The constraint $\sigma_c \leq 1$ directly means $\sigma_m \leq \sigma_y$.

**$\sigma_y = 40\,\text{MPa}$:** Much lower than the Al 6061-T6 tensile yield strength (≈ 250 MPa). This large safety factor accounts for dynamic loading (sudden gust loads on a bowsprit are impulsive, not quasi-static), fatigue, and uncertainty in load magnitude.

### 8f. p-continuation for stress — `p_stress_schedule`

**Source: Lecture 12, p. 4**

```
p = (2, 2, 4, 4, 8, 8, 16, 32, 64, 128, 200, 300, 300, 400, 400)
```

Synchronized with the 15 $\beta$-stages. Low $p$ first: the p-mean is smooth, gradients are well-conditioned, and the optimiser can move freely. As $p$ increases, the p-mean progressively approximates the true maximum-stress constraint. Starting at high $p$ gives noisy, hard-to-optimise gradients.

---

## 9. Volume Constraints — `volume_constraint`, `pitch_weighted_volume`

**Source: Lecture 8, p. 9 eq. (14) (dilated constraint)**

### Plain volume

$$g_V = \frac{1}{L H}\int_\Omega \bar{\rho}_d\,d\Omega - V_f \leq 0, \qquad V_f = 0.25$$

Applied on the **dilated** realization $\bar{\rho}_d$ (§5): if the most material-rich realization satisfies the volume bound, the nominal and eroded do so automatically.

### Pitch-weighted volume

A second volume constraint penalises material near the tip more strongly:

$$w(x) = 1 + \alpha\!\left(\frac{x}{L}\right)^2, \quad \alpha = 2.5$$

$$g_{V,w} = \frac{\int_\Omega w(x)\,\bar{\rho}_d\,d\Omega}{\int_\Omega w(x)\,d\Omega} - V_f \leq 0$$

The normalisation denominator equals $|\Omega|(1 + \alpha/3) = LH(1 + 2.5/3)$ so that a uniform density of $V_f$ exactly satisfies the constraint.

Physically: the bowsprit is cantilevered at the hull, so tip deflection drives compliance. Material near the tip ($x \approx L$) carries high weight $w \approx 1 + \alpha$, making it more "expensive" per unit volume. The optimiser is incentivised to route load-bearing material closer to the root where it is structurally efficient.

---

## 10. MMA Optimiser — `_make_mma`, `optimize`

**Source: Lecture 5**

### Method of Moving Asymptotes (MMA)

At each iteration MMA constructs convex separable approximations to all objective and constraint functions around the current point $\rho^{(k)}$, then minimises the approximate sub-problem with a simple bound-constrained solver. The approximations involve "asymptotes" $L_j^{(k)} < \rho_j < U_j^{(k)}$ that are updated adaptively.

### Parameter values

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `move` | 0.2 | Max design-variable change per iteration. 0.1–0.2 is standard; 0.2 allows faster early convergence while remaining stable. |
| `xtol` | 1e-4 | Convergence tolerance on design-variable change. |
| `ftol` | 1e-5 | Convergence tolerance on objective change. |
| `lmax` | 15 | Outer iterations per $\beta$-stage (in `__main__`; class default is 5). |
| `kmax` | 30 | Inner MMA iterations per outer iteration (in `__main__`; class default is 20). |

### Constraint vector ordering

MMA sees the function vector $\mathbf{f} = [J,\; g_V,\; g_{V,w},\; g_{S,\eta_0},\; g_{S,\eta_1},\; \ldots]$ where $J$ is the objective (index 0, minimised) and all subsequent entries are inequality constraints $\leq 0$.

---

## 11. Parallelisation — `ParallelBowspritOptimizer`, `parallel.py`

**Source: Lecture 8, p. 30–36**

### 11a. Why parallelise

Robust optimisation requires 3 FEM solves per iteration (one per $\eta$-realization), tripling the cost relative to non-robust optimisation (Lecture 8, p. 29–30). Solving the 3 instances simultaneously on 3 CPU groups recovers near-original wall-clock time:

> "Simple parallelisation using 3 CPUs: 34 minutes. Using parallel.py with 3 CPUs: 14 minutes."
> — Lecture 8, p. 35

### 11b. MPI communicator splitting (Lecture 8, p. 30–33)

`Parallel(MPI.comm_world, N)` splits the world communicator into $N$ groups (Lecture 8, p. 33):

```python
self.parallel = Parallel(MPI.comm_world, 3)  # 3 groups for 3 η realizations
group_comm   = self.parallel.group_comm       # communicator for this group
group_id     = self.parallel.group            # integer 0, 1, or 2
```

Lecture 8, p. 34 code assigns η by group:
- Group 0 → nominal, $\eta_N = 0.5$
- Group 1 → dilated, $\eta_D = \eta_N - \Delta\eta$
- Group 2 → eroded, $\eta_E = \eta_N + \Delta\eta$

Data communication primitives (Lecture 8, p. 33):

| Call | Direction | Purpose |
|------|-----------|---------|
| `fglobal2group(rho_world, rho_group)` | World → Group | broadcast design variables |
| `sgroup2global(J_group)` | Group → World | gather scalar cost/constraint values |
| `vgroup2global(dc_group)` | Group → World | gather gradient vectors |

### 11c. DOF mapping — `_build_dof_mapping` (Lecture 8, p. 32)

Lecture 8, p. 32: *"While the world mesh and group meshes are the same, FEniCS may arrange the DOFs differently. A mapping between the Group and World DOFs must be made."*

The lecture provides `parallel.py::create_mapping` which uses:

```python
idx = np.isin(a_point, b_point)
```

**Bug:** `np.isin(a, b)` returns a boolean mask over `a` (world DOF list order), not over `b` (group DOF list order). When the two communicators partition the mesh cells differently, the resulting index set is in world order, not group order → the mapping scrambles DOF values.

**Fix in `bowsprit_topopt.py`:** Use a KD-tree to match by physical coordinates, iterating in group-DOF order:

```python
from scipy.spatial import cKDTree
_, idx = cKDTree(coor_all).query(coor_group)   # idx[i] = world DOF nearest to group DOF i
```

For each group-local DOF $i$, `idx[i]` gives the corresponding world DOF index. Order follows the group side → correct bijection.

### 11d. Mesh creation on group communicator (Lecture 8, p. 32)

Lecture 8, p. 34 suggests: write mesh to XDMF, read it in each group. The bowsprit code instead calls:

```python
RectangleMesh.create(group_comm, ...)
```

This is necessary because reading from XDMF via `fa.Mesh` returns a raw C++ binding that is missing the `_ad_will_add_as_dependency` attribute required by `fenics_adjoint`. Constructing the mesh directly with the group communicator avoids this `AttributeError`.

### 11e. World vs group design variable

| Variable | Communicator | Purpose |
|----------|-------------|---------|
| `_rho_global` | world | used by MMA, which needs the full DOF vector |
| `_beam_group._rho` | group | used for group-local filter and FEM solves |

Before each FEM solve: `fglobal2group` copies the world $\rho$ into each group's local copy. After: `sgroup2global` / `vgroup2global` bring cost values and gradients back to world for MMA.

---

## Summary Table

| Code concept | Lecture | Page | Key value |
|---|---|---|---|
| Elasticity strong/weak form | 3 | 7–9 | — |
| Plane stress (not plane strain) | 3 | 7 | $\lambda_\text{eff}=E\nu/(1-\nu^2)$ |
| SIMP penalty | 6 | 9 | $p=3$ |
| SIMP void regularisation | 6 | 9 | $E_{min}=10^{-6}E_0$ |
| Helmholtz filter PDE | 6 | 12 | $r=0.04\,\text{m}$ |
| Filter ↔ convolution equivalence | 8 | 18 | $R=2\sqrt{3}r$ |
| Heaviside projection | 6 | 14; 8 p.7 | tanh formula |
| Robust 3-realization, volume on dilated | 8 | 9 eq. 14 | $\Delta\eta=0.05$ |
| Double-filter pipeline | 10 | 10, 16 | $r_2=r_1$ (deviation) |
| $\beta$-schedule | 6 | 17 | 1…432 (15 stages) |
| $\beta_\text{cap}$ | 11 | 9 | 4.8 ($n_y=30$) |
| $\varepsilon$-relaxation | 11 | 7 | $\varepsilon=0.2$ |
| Von Mises (plane-stress correction) | 11 | 7 | includes $s_{33}$ |
| p-mean stress constraint | 11 | 10–11 | $\sigma_y=40\,\text{MPa}$ |
| p-continuation | 12 | 4 | 2…400 (15 stages) |
| Parallelization / group splitting | 8 | 30–33 | 3 groups |
| DOF mapping (KDTree fix) | 8 | 32 | fixes `np.isin` bug |

---

## Code Comment Errors

| Location | Comment says | Correct source |
|---|---|---|
| `stresses_with_simp` | "Lecture 9/10" | Lecture 6, p. 9 (Lecture 9 is acoustic waves) |
| `solve_topopt` | "Lecture 3/4" | Lecture 3, p. 7–9 (Lecture 4 is constrained optimisation) |
| `beta_cap` comment | 9.6 | 4.8 used — matches $n_y=30$; 9.6 is correct for $n_y=60$ |
