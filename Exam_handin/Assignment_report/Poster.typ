// A3 two-column poster - topology optimization of a bowsprit spar
// Domain: L = 3.80 m × H = 0.50 m, plane-stress, two oblique tractions
// Exam 2026 Tasks 2-5: SIMP + Helmholtz filter + double Heaviside + MMA

#import "@preview/cetz:0.3.4": canvas, draw

// ── drawing parameters ───────────────────────────────────────────────────────
#let dom-L    = 10.0   // cetz units - proportional to L = 3.80 m
#let dom-H    = 1.32   // 10 × (0.50/3.80) → correct 7.6:1 aspect ratio
#let f1-angle = 65deg
#let f2-angle = 55deg
#let arr-len  = 1.10
#let arc-r    = 0.40

#set page(paper: "a3", margin: (x: 18mm, y: 18mm), columns: 2)
#set text(size: 10pt)
#set par(justify: true, leading: 0.50em)
#set figure(gap: 2pt)
#set math.equation(numbering: none)

#let section-head(body) = { v(4pt); text(weight: "bold", size: 11.5pt)[#body]; v(0pt) }
#let sub-head(body)     = { v(2pt); text(weight: "bold")[#body]; v(0pt) }

// ── title ─────────────────────────────────────────────────────────────────────

#place(top + center, scope: "parent", float: true)[
  #block(width: 100%, inset: (y: 1pt))[
    #align(center)[
      #grid(
        columns: (auto, 1fr, auto),
        align: (left + horizon, center + horizon, right + horizon),
        gutter: 8mm,
        image("images/au_logo.png", height: 20mm),
        [
          #text(weight: "bold", size: 22pt)[Topology optimized bowsprit for an 18-footer] \
          #v(3pt)
          #text(size: 10.5pt)[Sigurd Mousten Jager Nielsen · 202108107 · Aarhus Universitet]
        ],
        scale(x: -100%)[#image("images/18-footer.jpg", height: 26mm)],
      )
    ]
    //#v(0pt)
    #line(length: 100%, stroke: 0.6pt)
  ]
]

// ═══════════════════════════════════════════════════════════════════════════
// COLUMN 1
// ═══════════════════════════════════════════════════════════════════════════

#section-head[The situation]

An 18-footer skiff carries a large gennaker on a forward *bowsprit*. Rope tension and sail aerodynamic forces demand a spar that is stiff and light.

#figure(
  box(clip: true, width: 75%, height: 100mm)[
    #move(dy: -6mm)[
      #scale(x: -100%)[#image("images/18-footer.jpg", width: 100%)]
    ]
  ],
  caption: [18-footer skiff with gennaker. Bowsprit projects forward; sail tack attaches at tip. Photo: MySailing @18footer_photo.],
)

#section-head[Problem definition]

Modelled as a *2D plane-stress* rectangular cantilevered beam made from aluminum with the properties $E_0 = 70$ GPa, $nu = 0.33$,  thickness $t = 0.05$ m and minimum volume fraction $V_f = 0.25$. The domain is $3.80m times 0.50m #h(0.5em) (L times H)$ with the forces: $bold(F)_1 = 2$ kN at 65° on a small strip $0.175L$ to $0.225L$ and $bold(F)_2 = 3$ kN at 55° on a small strip $0.95L$ to $L$ as seen in @Beam_diagram. 

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

    // F2: strip 0.95L-1.0L; arrow at centre 0.975L, 55°
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
    Diagram of the beam with loads and boundary constraints.
  ],
)<Beam_diagram>

#section-head[Elasticity BVP - weak form]

Find $bold(u) in V_0$ such that $forall bold(v) in V_0$:

$ integral_Omega bold(sigma)(bold(u)) : bold(epsilon)(bold(v)) dif Omega = integral_(Gamma_N) bold(T) dot bold(v) dif Gamma $

Small-strain, plane-stress: $bold(sigma) = lambda_"eff" "tr"(bold(epsilon)) bold(I) + 2 mu bold(epsilon)$, $lambda_"eff" = frac(E nu, 1-nu^2)$. BC: $bold(u) = bold(0)$ on $Gamma_D$; tractions $bold(T)$ on $Gamma_N$.

*Compliance* (cost function) = external work: $J = integral_(Gamma_N) bold(T) dot bold(u) dif Gamma$

#section-head[Optimization problem]

Design variable $rho in [0,1]$ describing density element-wise on a mesh (DG0).

$
  min_(bold(rho)) cases(
    J_d + J_i + J_e & "Task 3",
    J_d + J_e        & "Task 4",
  ) quad "s.t." quad cases(
    frac(1, L H) integral_Omega overline(rho)_d dif Omega <= V_f,
    frac(integral_Omega w overline(rho)_d dif Omega, integral_Omega w dif Omega) <= V_f,
    sigma_c^k <= 1 quad forall k,
    0 <= rho_e <= 1,
  )
$

Pitch weight $w = 1 + 2.5(x/L)^2$ punishes mass towards the tip of the bowsprit. The constraints are evaluated on *dilated* $overline(rho)_d$ to guarantee all realizations satisfy limits.

// ═══════════════════════════════════════════════════════════════════════════
// COLUMN 2
// ═══════════════════════════════════════════════════════════════════════════

#section-head[SIMP - stiffness interpolation]

$ E_e = E_"min" + overline(rho)_e^p (E_0 - E_"min"), quad p = 3, quad E_"min" = 10^(-6) E_0 $

$p = 3$ penalises intermediate densities; $E_"min"$ prevents a singular stiffness matrix in voids.

#section-head[Double-filter robust formulation]

Three realizations from a single $rho$ via a two-stage pipeline:

$ rho arrow.r^"filter" tilde(rho)_1 arrow.r^(eta_"min") overline(rho)_1 arrow.r^"filter" tilde(rho)_2 arrow.r^(eta_k) overline(rho)_k $

Filters solved in CG1 (radius $r = 0.04$ m); projections in DG0.

#sub-head[Helmholtz PDE filter (weak form)]

$ integral_Omega r^2 nabla tilde(rho) dot nabla v dif Omega + integral_Omega tilde(rho)\, v dif Omega = integral_Omega rho\, v dif Omega quad forall v $

with $partial tilde(rho) slash partial n = 0$ on $partial Omega$. Enforces minimum feature size $approx 2r = 0.08$ m. CG1 trial space required (DG0 gradients vanish).

#sub-head[Heaviside projection]

$ overline(rho) = frac(tanh(beta eta) + tanh(beta(tilde(rho) - eta)), tanh(beta eta) + tanh(beta(1 - eta))) $

Sharpness $beta$ ramped $(1,2,...,432)$, 15 stages, 30 MMA iters/stage: $eta_d=0.45$ (*dilated*), $eta_i=0.50$ (*nominal*), $eta_e=0.55$ (*eroded*); $Delta eta = 0.05$. Shared load path = *manufacturing-robust* design.

#section-head[Stress constraint - P-mean aggregation]

Yield stress $sigma_y = 40$ MPa. One constraint per realization:

$ sigma_c^p = frac(1, |Omega|) integral_Omega (sigma_m / (alpha sigma_y))^p dif Omega <= 1 $

$epsilon$-relaxation removes void singularity ($epsilon = 0.2$):

$ f_sigma = frac(overline(rho), epsilon(1 - overline(rho)) + overline(rho)), quad sigma_m = f_sigma sqrt(sigma_"vM"^2 + sigma_"min"^2) $

$ sigma_"vM"^2 = sigma_11^2 + sigma_22^2 - sigma_11 sigma_22 + 3 sigma_12^2 $

*$beta$ cap* $beta <= 2R\/l_e approx 9.6$ prevents artificial stress concentrations. *$p$-continuation*: $p$ ramped $2 arrow.r 400$ over 15 stages.

#section-head[Optimized designs]

#figure(
  grid(
    columns: (1fr,),
    gutter: 4mm,
    [
      #text(size: 8pt, weight: "bold")[Task 3 (3 realizations)]
      #image("images/Task_3.png", width: 100%)
    ],
    [
      #text(size: 8pt, weight: "bold")[Task 4 (2 realizations)]
      #image("images/Task_4.png", width: 100%)
    ],
  ),
  caption: [Optimized density fields. Top to bottom per task: eroded ($eta_e=0.55$), nominal ($eta_i=0.50$, Task 3 only), dilated ($eta_d=0.45$). Blue = solid, white = void.],
)

#section-head[Results & conclusion (Task 5)]

#text(size: 8.5pt)[
#table(
  columns: (2.0fr, 1.0fr, 0.7fr, 0.7fr, 1.0fr),
  align: (left, right, right, right, right),
  inset: (x: 4pt, y: 3pt),
  table.header([*Design*], [*J* (N·m)], [*V*], [*V*#sub[p]], [*σ*#sub[c]]),
  [T3 dilated],  [10.09], [.299#super[†]], [.291#super[†]], [0.619],
  [T3 nominal],  [11.65], [.257#super[†]], [.248#super[†]], [0.664],
  [T3 eroded],   [14.82], [.204],         [.193],          [*1.126*#super[†]],
  [T4 dilated],  [10.51], [.301#super[†]], [.295#super[†]], [0.654],
  [T4 interm.],  [12.03], [.260#super[†]], [.253#super[†]], [0.729],
  [T4 eroded],   [15.41], [.207],         [.199],          [*1.152*#super[†]],
)
#super[†]constraint violated ($V_f = 0.25$, $sigma_c <= 1$). Task 3 nominal J is 3.3% lower than Task 4 intermediate - including the nominal realization yields a stiffer design. SIMP, Helmholtz filtering, and double Heaviside projection produce a lightweight bowsprit; stress constraints remain binding in the eroded realization.
]

#line(length: 100%, stroke: 0.4pt)
#bibliography("refs.bib", title: none, style: "ieee")
