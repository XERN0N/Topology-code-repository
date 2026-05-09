// A3 two-column poster — topology optimization of a bowsprit spar
// Domain: L = 3.80 m × H = 0.50 m, plane-strain, two oblique tractions
// Exam 2026 Tasks 2–5: SIMP + Helmholtz filter + double Heaviside + MMA

#import "@preview/cetz:0.3.4": canvas, draw

// ── drawing parameters ───────────────────────────────────────────────────────
#let dom-L    = 10.0   // cetz units — proportional to L = 3.80 m
#let dom-H    = 1.32   // 10 × (0.50/3.80) → correct 7.6:1 aspect ratio
#let f1-angle = 65deg
#let f2-angle = 55deg
#let arr-len  = 1.10
#let arc-r    = 0.40

#set page(paper: "a3", margin: (x: 18mm, y: 18mm), columns: 2)
#set text(size: 10pt)
#set par(justify: true, leading: 0.56em)
#set figure(gap: 3pt)
#set math.equation(numbering: none)

#let section-head(body) = { v(5pt); text(weight: "bold", size: 11.5pt)[#body]; v(1pt) }
#let sub-head(body)     = { v(3pt); text(weight: "bold")[#body]; v(0pt) }

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

An 18-footer skiff uses a fixed *bowsprit* — a spar projecting forward from the bow to carry a large gennaker. Rope tension and sail aerodynamic forces create a combined axial and transverse load. The spar must be as stiff and light as possible.

#figure(
  box(clip: true, width: 80%, height: 110mm)[
    #move(dy: -6mm)[
      #scale(x: -100%)[#image("images/18-footer.jpg", width: 100%)]
    ]
  ],
  caption: [18-footer skiff with gennaker. Bowsprit projects forward; sail tack attaches at tip. Photo: MySailing @18footer_photo.],
)

#section-head[Problem definition]

The spar cross-section is constant, dominant loading in-plane. Modelled as a *2D plane-strain* rectangular cantilever fixed at the deck fitting. $E_0 = 70$ GPa, $nu = 0.33$, $t = 0.05$ m, volume fraction limit $V_f = 0.25$.

#figure(
  canvas(length: 0.87cm, {
    import draw: *
    rect((0, 0), (dom-L, dom-H))
    line((0, 0), (0, dom-H), stroke: 2.5pt)
    for i in range(6) {
      let y = dom-H * i / 7
      line((-0.38, y), (0.0, y + 0.32), stroke: 0.5pt)
    }
    content((dom-L / 2, -0.52), text(size: 6pt)[$L = 3.80$ m])
    content((-1.08, dom-H / 2), text(size: 6pt)[$H = 0.50$ m])
    content((-0.6, -0.32), text(size: 5.5pt, fill: gray)[Fixed])

    // F1: strip centred at 0.20L, 65° from horizontal
    let x1     = dom-L * 0.20
    let a1-dir = 180deg - f1-angle
    line((x1, dom-H),
         (x1 - arr-len * calc.cos(f1-angle), dom-H + arr-len * calc.sin(f1-angle)),
         stroke: 1.3pt, mark: (end: ">"))
    arc((x1, dom-H), start: a1-dir, stop: 180deg, radius: arc-r, anchor: "origin", stroke: 0.5pt)
    let lbl-r  = arc-r + 0.35
    let a1-mid = (a1-dir + 180deg) / 2
    content((x1 + lbl-r * calc.cos(a1-mid), dom-H + lbl-r * calc.sin(a1-mid)),
            text(size: 5.5pt)[65°])
    content((x1 - arr-len * calc.cos(f1-angle), dom-H + arr-len * calc.sin(f1-angle) + 0.16),
            text(size: 6pt)[$bold(F)_1$])

    // F2: strip 0.95L–1.0L; arrow at centre 0.975L, 55°
    let x2     = dom-L * 0.975
    let a2-dir = 180deg - f2-angle
    line((x2, dom-H),
         (x2 - arr-len * calc.cos(f2-angle), dom-H + arr-len * calc.sin(f2-angle)),
         stroke: 1.3pt, mark: (end: ">"))
    arc((x2, dom-H), start: a2-dir, stop: 180deg, radius: arc-r, anchor: "origin", stroke: 0.5pt)
    let a2-mid = (a2-dir + 180deg) / 2
    content((x2 + lbl-r * calc.cos(a2-mid), dom-H + lbl-r * calc.sin(a2-mid)),
            text(size: 5.5pt)[55°])
    content((x2 - arr-len * calc.cos(f2-angle), dom-H + arr-len * calc.sin(f2-angle) + 0.16),
            text(size: 6pt)[$bold(F)_2$])
  }),
  caption: [
    Domain ($L times H = 3.80 times 0.50$ m). $bold(F)_1 = 2$ kN at 65° on strip $0.175L$–$0.225L$; $bold(F)_2 = 3$ kN at 55° on strip $0.95L$–$L$. Upper-left direction.
  ],
)

#section-head[Elasticity BVP — weak form]

Find $bold(u) in V_0$ such that $forall bold(v) in V_0$:

$ integral_Omega bold(sigma)(bold(u)) : bold(epsilon)(bold(v)) dif Omega = integral_(Gamma_N) bold(T) dot bold(v) dif Gamma $

Small-strain: $bold(epsilon) = frac(1,2)(nabla bold(u) + nabla bold(u)^T)$. Plane-strain stress: $bold(sigma) = lambda "tr"(bold(epsilon)) bold(I) + 2 mu bold(epsilon)$, with $lambda = frac(E nu,(1+nu)(1-2nu))$, $mu = frac(E,2(1+nu))$. BC: $bold(u) = bold(0)$ on $Gamma_D$ (left edge); tractions $bold(T)$ on $Gamma_N = Gamma_"F1" union Gamma_"F2"$.

*Compliance* (cost function) = external work: $J = integral_(Gamma_N) bold(T) dot bold(u) dif Gamma$

#section-head[Optimization problem]

Design variable $rho in [0,1]$ element-wise (DG0, one DOF/cell). *Task 3*: $min_(bold(rho)) J_d + J_i + J_e$. *Task 4*: $min_(bold(rho)) J_d + J_e$ ($J_i$ post-hoc). Both subject to:

$ V(overline(rho)_d) = frac(integral_Omega overline(rho)_d dif Omega, L H) <= V_f $

$ integral_Omega w(x)\, overline(rho)_d dif Omega <= V_f integral_Omega w(x) dif Omega $

with pitch weight $w(x) = 1 + 2.5(x/L)^2$ and $0 <= rho_e <= 1$. Constraints on *dilated* $overline(rho)_d$ (most material) guarantee all realizations satisfy limits. Gradients via pyadjoint (adjoint method); solved by MMA.

// ═══════════════════════════════════════════════════════════════════════════
// COLUMN 2
// ═══════════════════════════════════════════════════════════════════════════

#section-head[SIMP — stiffness interpolation]

$ E_e = E_"min" + overline(rho)_e^p (E_0 - E_"min"), quad p = 3, quad E_"min" = 10^(-6) E_0 $

Penalty $p = 3$: intermediate densities are costly ($overline(rho)=0.5$ gives only $12.5%$ of $E_0$), driving black–white designs. $E_"min"$ prevents a singular stiffness matrix in void regions.

#section-head[Double-filter robust formulation]

Small manufacturing tolerances can erode or dilate thin members. Robustness is enforced by producing three physical density realizations from a single $rho$ through a two-stage filter–project pipeline:

- *Helmholtz filter* $rho arrow.r tilde(rho)_1$ (solved in CG1, radius $r = 0.04$ m)
- *Heaviside project* $tilde(rho)_1 arrow.r overline(rho)_1$ at fixed $eta_"min" = 0.45$ (DG0)
- *Helmholtz filter* $overline(rho)_1 arrow.r tilde(rho)_2$ again (CG1)
- *Heaviside project* $tilde(rho)_2 arrow.r overline(rho)_k$ at realization threshold $eta_k$ (DG0)

#sub-head[Helmholtz PDE filter (weak form)]

$ integral_Omega r^2 nabla tilde(rho) dot nabla v dif Omega + integral_Omega tilde(rho)\, v dif Omega = integral_Omega rho\, v dif Omega quad forall v $

with $partial tilde(rho) slash partial n = 0$ on $partial Omega$. Enforces minimum feature size $approx 2r = 0.08$ m. Requires CG1 trial space — DG0 gradients vanish inside cells, making the diffusion term zero.

#sub-head[Heaviside projection]

$ overline(rho) = frac(tanh(beta eta) + tanh(beta(tilde(rho) - eta)), tanh(beta eta) + tanh(beta(1 - eta))) $

Sharpness $beta$ ramped via *beta continuation* $\{1,2,4,...,256\}$ (40 MMA iters/stage):

#figure(
  table(
    columns: (1fr, 1fr, 1fr),
    align: center,
    table.header[$eta_d = 0.45$][$eta_i = 0.50$][$eta_e = 0.55$],
    [*Dilated*], [*Nominal*], [*Eroded*],
  ),
  caption: [Three realizations ($Delta eta = 0.05$). Same $rho$, same pipeline, different $eta_k$.],
)

If all three share the same load-path topology, the design is *manufacturing-robust*. *Task 3* includes $J_i$ in the objective directly; *Task 4* excludes it, computing the intermediate design post-hoc from the final $rho$ at the final $beta$.

#section-head[Optimized design]

#figure(
  box(clip: true, width: 100%, height: 38mm)[
    #move(dy: -40mm)[
      #image("images/rho_bar_task3.png", width: 100%, height: 80mm)
    ]
  ],
  caption: [Nominal optimized bowsprit ($eta_i = 0.50$, Task 3). Blue = solid, white = void.],
)

#section-head[Results and comparison (Task 5)]

#figure(
  table(
    columns: (1.7fr, 1fr, 0.8fr, 0.9fr),
    align: (left, right, right, right),
    table.header[*Design*][*J* (N·m)][*V*][*V*#sub[pitch]],
    [Task 3 dilated],      [—], [—], [—],
    [Task 3 nominal],      [—], [—], [—],
    [Task 3 eroded],       [—], [—], [—],
    [Task 4 dilated],      [—], [—], [—],
    [Task 4 intermediate], [—], [—], [—],
    [Task 4 eroded],       [—], [—], [—],
  ),
  caption: [Compliance $J$ and volume fractions. Target: $V <= 0.25$.],
)

Task 3 minimizes nominal compliance directly, favouring a stiffer intermediate design. Task 4 may achieve lower $J_d + J_e$ (fewer terms to balance) but the intermediate realization is never optimized. For this slender spar where bending dominates, small member-width changes strongly affect compliance — including the nominal design (Task 3) is likely preferable.

#section-head[Conclusion]

Topology optimization with SIMP, Helmholtz PDE filtering, and double Heaviside projection yields a lightweight bowsprit within a 25% volume budget. Three-realization (Task 3) vs two-realization (Task 4) robust optimization quantifies the trade-off between nominal compliance and solver cost per iteration.

#v(1fr)
#line(length: 100%, stroke: 0.4pt)
#v(2pt)
#text(size: 8pt)[Topology optimization · Exam hand-in · Aarhus Universitet · 2026]

#bibliography("refs.bib", title: none, style: "ieee")
