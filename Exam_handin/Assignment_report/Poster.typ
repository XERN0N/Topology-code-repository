// A3 two-column poster — topology optimization of a bowsprit spar
// Domain: L = 3.80 m × H = 0.50 m, plane-strain, two oblique tractions
// Exam 2026 Tasks 2–5: SIMP + Helmholtz filter + double Heaviside + MMA

#import "@preview/cetz:0.3.4": canvas, draw

// ── drawing parameters ───────────────────────────────────────────────────────
#let dom-L    = 10.0   // cetz units — proportional to L = 3.80 m
#let dom-H    = 1.32   // 10 × (0.50/3.80) ≈ 1.316 → correct 7.6:1 aspect ratio
#let f1-angle = 65deg  // F1 from horizontal
#let f2-angle = 55deg  // F2 from horizontal
#let arr-len  = 1.10   // arrow shaft in cetz units
#let arc-r    = 0.40   // angle-arc radius

#set page(paper: "a3", margin: (x: 18mm, y: 18mm), columns: 2)
#set text(size: 10pt)
#set par(justify: true, leading: 0.58em)
#set figure(gap: 5pt)
#set math.equation(numbering: none)

#let section-head(body) = { v(7pt); text(weight: "bold", size: 11.5pt)[#body]; v(2pt) }
#let sub-head(body)     = { v(4pt); text(weight: "bold")[#body]; v(1pt) }

// ── title ─────────────────────────────────────────────────────────────────────

#place(top + center, scope: "parent", float: true)[
  #block(width: 100%, inset: (y: 5pt))[
    #align(center)[
      #grid(
        columns: (auto, 1fr, auto),
        align: (left + horizon, center + horizon, right + horizon),
        gutter: 8mm,
        image("images/au_logo.png", height: 26mm),
        [
          #text(weight: "bold", size: 22pt)[Topology optimized bowsprit for an 18-footer dinghy] \
          #v(3pt)
          #text(size: 10.5pt)[Sigurd Mousten Jager Nielsen · 202108107 · Aarhus Universitet]
        ],
        scale(x: -100%)[#image("images/18-footer.jpg", height: 26mm)],
      )
    ]
    #v(4pt)
    #line(length: 100%, stroke: 0.6pt)
  ]
]

// ═══════════════════════════════════════════════════════════════════════════
// COLUMN 1
// ═══════════════════════════════════════════════════════════════════════════

#section-head[The situation]

An 18-footer skiff uses a fixed *bowsprit* — a spar projecting forward from the bow to carry a large gennaker sail. Rope tension and sail aerodynamic forces create a combined axial and transverse load. The spar must be as stiff and light as possible: excess weight at the bow hurts trim, while insufficient stiffness causes the luff to sag and lose sail power.

#figure(
  scale(x: -100%)[#image("images/18-footer.jpg", width: 100%)],
  caption: [18-footer skiff with gennaker set. The bowsprit projects forward; the sail tack attaches at its tip. Photo: MySailing @18footer_photo.],
)

#section-head[Problem definition]

The bowsprit cross-section is constant, so the dominant loading is in-plane. The spar is modelled as a *2D plane-strain* rectangular cantilever fixed at the deck fitting.

#figure(
  canvas(length: 0.87cm, {
    import draw: *

    // domain rectangle
    rect((0, 0), (dom-L, dom-H))

    // clamped left edge: thick bar + hatch marks
    line((0, 0), (0, dom-H), stroke: 2.5pt)
    for i in range(6) {
      let y = dom-H * i / 7
      line((-0.38, y), (0.0, y + 0.32), stroke: 0.5pt)
    }

    // dimension labels
    content((dom-L / 2, -0.52), text(size: 6pt)[$L = 3.80$ m])
    content((-1.08, dom-H / 2), text(size: 6pt)[$H = 0.50$ m])
    content((-0.6, -0.32), text(size: 5.5pt, fill: gray)[Fixed])

    // F1: strip centred at 0.20L (strip 0.175L–0.225L), upper-left at 65°
    let x1    = dom-L * 0.20
    let a1-dir = 180deg - f1-angle
    line(
      (x1, dom-H),
      (x1 - arr-len * calc.cos(f1-angle), dom-H + arr-len * calc.sin(f1-angle)),
      stroke: 1.3pt, mark: (end: ">"),
    )
    arc((x1, dom-H), start: a1-dir, stop: 180deg, radius: arc-r, anchor: "origin", stroke: 0.5pt)
    let lbl-r  = arc-r + 0.35
    let a1-mid = (a1-dir + 180deg) / 2
    content(
      (x1 + lbl-r * calc.cos(a1-mid), dom-H + lbl-r * calc.sin(a1-mid)),
      text(size: 5.5pt)[65°],
    )
    content(
      (x1 - arr-len * calc.cos(f1-angle), dom-H + arr-len * calc.sin(f1-angle) + 0.16),
      text(size: 6pt)[$bold(F)_1$],
    )

    // F2: strip 0.95L–1.0L; arrow at strip centre 0.975L, upper-left at 55°
    let x2    = dom-L * 0.975
    let a2-dir = 180deg - f2-angle
    line(
      (x2, dom-H),
      (x2 - arr-len * calc.cos(f2-angle), dom-H + arr-len * calc.sin(f2-angle)),
      stroke: 1.3pt, mark: (end: ">"),
    )
    arc((x2, dom-H), start: a2-dir, stop: 180deg, radius: arc-r, anchor: "origin", stroke: 0.5pt)
    let a2-mid = (a2-dir + 180deg) / 2
    content(
      (x2 + lbl-r * calc.cos(a2-mid), dom-H + lbl-r * calc.sin(a2-mid)),
      text(size: 5.5pt)[55°],
    )
    content(
      (x2 - arr-len * calc.cos(f2-angle), dom-H + arr-len * calc.sin(f2-angle) + 0.16),
      text(size: 6pt)[$bold(F)_2$],
    )
  }),
  caption: [
    Design domain ($L times H = 3.80 times 0.50$ m, $t = 0.05$ m). Left edge clamped. $bold(F)_1 = 2$ kN at 65° on top-edge strip at $0.175L$–$0.225L$; $bold(F)_2 = 3$ kN at 55° on strip $0.95L$–$L$. Both tractions directed upper-left.
  ],
)

Material and constraints: $E_0 = 70$ GPa, $nu = 0.33$, volume fraction limit $V_f = 0.25$.

#section-head[Elasticity BVP — weak form]

Find $bold(u) in V_0$ such that $forall bold(v) in V_0$:

$ integral_Omega bold(sigma)(bold(u)) : bold(epsilon)(bold(v)) dif Omega = integral_(Gamma_N) bold(T) dot bold(v) dif Gamma $

Small-strain tensor: $bold(epsilon)(bold(u)) = frac(1,2)(nabla bold(u) + nabla bold(u)^T)$.
Plane-strain stress: $bold(sigma) = lambda "tr"(bold(epsilon)) bold(I) + 2 mu bold(epsilon)$.
Lamé constants: $lambda = frac(E nu,(1+nu)(1-2nu))$, $mu = frac(E,2(1+nu))$.
$bold(u) = bold(0)$ on clamped left edge $Gamma_D$; prescribed tractions $bold(T)$ on $Gamma_N = Gamma_"F1" union Gamma_"F2"$.

*Compliance* (external work, cost function):
$ J = integral_(Gamma_N) bold(T) dot bold(u) dif Gamma $

#section-head[Optimization problem]

Design variable $rho in [0,1]$ is defined element-wise (one DOF per mesh cell).

*Task 3* — optimize three realizations simultaneously:
$ min_(bold(rho)) quad J_d (bold(rho)) + J_i (bold(rho)) + J_e (bold(rho)) $

*Task 4* — optimize dilated and eroded only:
$ min_(bold(rho)) quad J_d (bold(rho)) + J_e (bold(rho)) $

Both subject to (constraints on *dilated* realization $overline(rho)_d$, most material):
$ V(overline(rho)_d) := frac(integral_Omega overline(rho)_d dif Omega, L H) <= V_f $
$ V_"pitch"(overline(rho)_d) := frac(integral_Omega w(x) overline(rho)_d dif Omega, integral_Omega w(x) dif Omega) <= V_f $
$ 0 <= rho_e <= 1 quad forall e $

Pitch weight $w(x) = 1 + 2.5(x/L)^2$ penalizes material near the tip where pitching inertia is greatest. Using the dilated realization for constraints guarantees all three designs satisfy the volume limit.

// ═══════════════════════════════════════════════════════════════════════════
// COLUMN 2
// ═══════════════════════════════════════════════════════════════════════════

#section-head[SIMP — stiffness interpolation]

Each element's Young's modulus is penalized by its physical density $overline(rho)_e$:

$ E_e = E_"min" + overline(rho)_e^p (E_0 - E_"min"), quad p = 3, quad E_"min" = 10^(-6) E_0 $

Penalty $p = 3$ makes intermediate densities structurally costly: $overline(rho)_e = 0.5$ contributes only $0.5^3 = 12.5%$ of $E_0$, driving the design toward black–white (solid–void). $E_"min"$ prevents a singular stiffness matrix in void regions. Gradients of $J$ with respect to $rho$ are computed automatically by pyadjoint (adjoint method).

#section-head[Double-filter robust formulation]

Manufacturing tolerances may erode or dilate thin members. Robustness is enforced by producing *three physical density realizations* from a single design variable field $rho$ through a two-stage filter–project pipeline applied twice:

#figure(
  table(
    columns: (auto, 1fr),
    stroke: none,
    align: (right, left),
    [$rho$],              [design variable (DG0, one DOF/cell)],
    [$arrow.b$],          [*Helmholtz filter*, radius $r = 0.04$ m],
    [$tilde(rho)_1$],     [smoothed field (CG1)],
    [$arrow.b$],          [*Heaviside project*, $eta_"min" = 0.45$],
    [$overline(rho)_1$],  [pre-projected density (DG0)],
    [$arrow.b$],          [*Helmholtz filter*, radius $r$],
    [$tilde(rho)_2$],     [re-smoothed field (CG1)],
    [$arrow.b$],          [*Heaviside project*, realization threshold $eta_k$],
    [$overline(rho)_k$],  [physical density realization (DG0)],
  ),
  caption: [Double-filter pipeline producing one realization $overline(rho)_k$.],
)

#sub-head[Helmholtz PDE filter]

Smoothing is performed by solving the elliptic PDE (weak form) in a CG1 space:

$ integral_Omega r^2 nabla tilde(rho) dot nabla v dif Omega + integral_Omega tilde(rho) v dif Omega = integral_Omega rho v dif Omega quad forall v $

with $partial tilde(rho)/partial n = 0$ on $partial Omega$. This removes checkerboard instabilities and enforces a minimum feature size $approx 2r = 0.08$ m. A CG1 (piecewise-linear) trial space is required: DG0 functions have zero gradient inside each element, collapsing the diffusion term to zero and reducing the filter to an identity.

#sub-head[Heaviside projection]

$ overline(rho) = frac(tanh(beta eta) + tanh(beta(tilde(rho) - eta)), tanh(beta eta) + tanh(beta(1 - eta))) $

Threshold $eta$ sets the realization type; sharpness $beta$ is ramped via *beta continuation* through $\{1,2,4,8,16,32,64,128,256\}$ with up to 40 MMA iterations per stage:

#figure(
  table(
    columns: (auto, auto, auto),
    align: center,
    table.header[$eta_d = 0.45$][$eta_i = 0.50$][$eta_e = 0.55$],
    [*Dilated*], [*Nominal*], [*Eroded*],
    [wider members], [balanced], [thinner members],
  ),
  caption: [Three robust realizations ($Delta eta = 0.05$). All share the same $rho$ and pipeline; only $eta_k$ differs.],
)

If all three realizations share the same load-path topology and differ only in member width, the design is *robust*: small manufacturing deviations do not change structural behavior.

*Task 3* optimizes $J_d + J_i + J_e$: the nominal design participates in the optimization, which constrains its compliance directly.
*Task 4* optimizes $J_d + J_e$ only: the intermediate design ($eta = 0.5$) is computed post-hoc from the final $rho$ at the final $beta$, without re-optimization.

#section-head[Optimized design]

#figure(
  box(clip: true, width: 100%, height: 38mm)[
    #move(dy: -40mm)[
      #image("images/rho_bar_task3.png", width: 100%, height: 80mm)
    ]
  ],
  caption: [Nominal optimized bowsprit topology ($eta_i = 0.50$, Task 3). Blue = solid, white = void.],
)

#section-head[Results and comparison (Task 5)]

#figure(
  table(
    columns: (1.7fr, 1fr, 0.9fr, 1fr),
    align: (left, right, right, right),
    table.header[*Design*][*J* (N·m)][*V*][*V*#sub[pitch]],
    [Task 3 dilated],       [—], [—], [—],
    [Task 3 nominal],       [—], [—], [—],
    [Task 3 eroded],        [—], [—], [—],
    [Task 4 dilated],       [—], [—], [—],
    [Task 4 intermediate],  [—], [—], [—],
    [Task 4 eroded],        [—], [—], [—],
  ),
  caption: [Compliance $J$ and volume fractions. Target: $V <= V_f = 0.25$.],
)

*Task 5 — which is better?*
Task 3 includes the nominal design in the objective, so the optimizer directly minimizes its compliance. Task 4 may achieve lower combined $J_d + J_e$ compliance (two rather than three terms to balance), but the intermediate design is never optimized and may show higher compliance. For a slender spar with dominant bending, including the nominal design (Task 3) is likely preferable: bending compliance is sensitive to member widths, and small erosion or dilation can significantly affect performance.

#section-head[Conclusion]

Topology optimization with SIMP, Helmholtz PDE filtering, and robust double Heaviside projection yields a lightweight bowsprit design within a 25% volume budget. The double-filter pipeline ensures that dilated, nominal, and eroded designs share the same topology, providing manufacturing robustness. Comparing Task 3 (three realizations) and Task 4 (two realizations) quantifies the trade-off between nominal compliance and optimization cost per iteration.

#v(1fr)
#line(length: 100%, stroke: 0.4pt)
#v(2pt)
#text(size: 8pt)[Topology optimization · Exam hand-in · Aarhus Universitet · 2026]

#bibliography("refs.bib", title: none, style: "ieee")
