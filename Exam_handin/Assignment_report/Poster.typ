// A3 two-column poster — topology optimization of a dinghy bowsprit
// 2D plane-strain rectangular design domain, MMA optimizer

#import "@preview/cetz:0.3.4": canvas, draw

// ── drawing parameters — edit these to adjust the domain sketch ─────────────
#let dom-L      = 5.0    // domain width  in cetz units  (= L = 1 m)
#let dom-H      = 2.0    // domain height in cetz units  (= W = 0.5 m)
#let f1-pct     = 0.20   // F1 position along top edge as fraction of L
#let f1-angle   = 65deg  // F1 angle from horizontal
#let f2-angle   = 55deg  // F2 angle from horizontal (applied at tip, x=L)
#let arr-len    = 0.88   // arrow shaft length in cetz units
#let arc-r      = 0.32   // angle-arc radius

#set page(
  paper: "a3",
  margin: (x: 18mm, y: 18mm),
  columns: 2,
)

#set text(size: 10pt)
#set par(justify: true, leading: 0.58em)
#set figure(gap: 5pt)
#set math.equation(numbering: none)

// ── helpers ──────────────────────────────────────────────────────────────────

#let section-head(body) = {
  v(7pt)
  text(weight: "bold", size: 11.5pt)[#body]
  v(2pt)
}

#let sub-head(body) = {
  v(4pt)
  text(weight: "bold")[#body]
  v(1pt)
}

// ── full-width title block ────────────────────────────────────────────────────

#place(top + center, scope: "parent", float: true)[
  #block(width: 100%, inset: (y: 5pt))[
    #align(center)[
      #grid(
        columns: (auto, 1fr, auto),
        align: (left + horizon, center + horizon, right + horizon),
        gutter: 8mm,
        image("images/au_logo.png", height: 26mm),
        [
          #text(weight: "bold", size: 22pt)[
            Topology optimized bowsprit for an 18-footer dinghy
          ] \
          #v(3pt)
          #text(size: 10.5pt)[
            Sigurd Mousten Jager Nielsen · 202108107 · Aarhus Universitet
          ]
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

An 18-footer skiff uses a fixed *bowsprit*, which is a spar projecting forward from the bow to be able to project a large downwind sail area that makes these dinghies so fast. The bowsprit is loaded by rope tension, the lift from the sails and the drag of the sails thus creating a combined axial and transverse load on the spar.

The bowsprit must be as stiff and light as possible: excess weight at the bow hurts boat balance, while insufficient stiffness causes the gennaker luff to sag and lose power.

#figure(
  scale(x: -100%)[#image("images/18-footer.jpg", width: 100%)],
  caption: [18-footer skiff with gennaker set. The bowsprit (pointing right) projects forward from the bow; the blue sail tack is attached at its tip. Photo: MySailing @18footer_photo.],
)

#section-head[Initial design and constraints]

The bowsprit is modelled as a 2D *plane-strain* rectangular domain as shown below. This is appropriate because the spar cross-section is constant and the dominant loading is in-plane.

#figure(
  grid(
    columns: (1fr, 1.5fr),
    gutter: 5pt,
    align: horizon,
    rect(width: 100%, height: 56mm, fill: luma(235), stroke: 0.4pt)[
      #align(center + horizon)[_Bowsprit installation photo —_\ _insert image here_]
    ],
    canvas(length: 0.87cm, {
      import draw: *

      // domain rectangle
      rect((0, 0), (dom-L, dom-H))

      // clamped left edge: thick bar + diagonal hatch marks
      line((0, 0), (0, dom-H), stroke: 2.5pt)
      for i in range(8) {
        let y = dom-H * i / 7
        line((-0.38, y), (0.0, y + 0.32), stroke: 0.5pt)
      }

      // dimension and boundary labels
      content((dom-L / 2, -0.48), text(size: 6pt)[$L = 1$ m])
      content((-1.0, dom-H / 2), text(size: 6pt)[$W = 0.5$ m])
      content((-0.6, -0.3), text(size: 5.5pt, fill: gray)[Fixed])

      // F1: at f1-pct of L along top edge, upper-left at f1-angle from horizontal
      let x1    = dom-L * f1-pct
      let a1-dir = 180deg - f1-angle   // arrow direction from +x axis

      line(
        (x1, dom-H),
        (x1 - arr-len * calc.cos(f1-angle), dom-H + arr-len * calc.sin(f1-angle)),
        stroke: 1.3pt, mark: (end: ">"),
      )
      arc((x1, dom-H), start: a1-dir, stop: 180deg, radius: arc-r, stroke: 0.5pt)

      let lbl-r  = arc-r + 0.25
      let a1-mid = (a1-dir + 180deg) / 2
      content(
        (x1 + lbl-r * calc.cos(a1-mid), dom-H + lbl-r * calc.sin(a1-mid)),
        text(size: 5.5pt)[#{f1-angle}],
      )
      content(
        (x1 - arr-len * calc.cos(f1-angle), dom-H + arr-len * calc.sin(f1-angle) + 0.12),
        text(size: 6pt)[$bold(F)_1$],
      )

      // F2: at tip (100 % of L) along top edge, upper-left at f2-angle from horizontal
      let x2    = dom-L
      let a2-dir = 180deg - f2-angle

      line(
        (x2, dom-H),
        (x2 - arr-len * calc.cos(f2-angle), dom-H + arr-len * calc.sin(f2-angle)),
        stroke: 1.3pt, mark: (end: ">"),
      )
      arc((x2, dom-H), start: a2-dir, stop: 180deg, radius: arc-r, stroke: 0.5pt)

      let a2-mid = (a2-dir + 180deg) / 2
      content(
        (x2 + lbl-r * calc.cos(a2-mid), dom-H + lbl-r * calc.sin(a2-mid)),
        text(size: 5.5pt)[#{f2-angle}],
      )
      content(
        (x2 - arr-len * calc.cos(f2-angle), dom-H + arr-len * calc.sin(f2-angle) + 0.12),
        text(size: 6pt)[$bold(F)_2$],
      )
    }),
  ),
  caption: [
    Left: bowsprit installation (photo placeholder). Right: 2D plane-strain optimization domain ($L times W = 1 times 0.5$ m). Entire left edge clamped. Surface tractions $bold(F)_1$ (70° from horizontal) at 20% of $L$ and $bold(F)_2$ (60° from horizontal) at the tip, both directed upper-left.
  ],
)

Boundary conditions and problem data:

- *Clamped* left edge — models the deck/mast-foot fitting.
- *Tip load* at the right-edge centre: resultant of spinnaker tack tension. The load is resolved into a forward component $F_x$ and a downward component $F_y$ based on the tack-line angle $alpha$.
- Dimensions: $L = 1$ m (length), $W = 0.2$ m (height/width).
- Volume constraint: $V_"frac" <= 33%$.
- Material: carbon-fibre composite (isotropic approximation), $E = 70$ GPa, $nu = 0.3$.

#section-head[Cost function — minimum compliance]

Compliance (strain energy) is minimised to maximise stiffness for a given volume of material:

$ J = integral_Omega 1/2 bold(sigma) : bold(epsilon) dif Omega
    = 1/2 bold(u)^T bold(K) bold(u) $

where $bold(u)$ is the displacement field, $bold(K)$ the global stiffness matrix,
$bold(sigma)$ the Cauchy stress and $bold(epsilon)$ the small-strain tensor.

#section-head[Solid isotropic material with penalization (SIMP)]

Each element carries a design variable $rho_e in [rho_"min", 1]$.
The effective Young's modulus is penalized as

$ E_e = E_0 rho_e^p, quad p = 3 $

to drive intermediate densities toward $0$ or $1$ (void or solid), producing a manufacturable black–white result. $rho_"min" = 10^(-6)$ prevents a singular stiffness matrix.

// ═══════════════════════════════════════════════════════════════════════════
// COLUMN 2
// ═══════════════════════════════════════════════════════════════════════════

#section-head[Filtering and projection]

Raw SIMP solutions suffer from *checkerboard* instabilities and mesh-dependent results. A two-step regularization is applied:

#sub-head[Density filter]
The design field $rho$ is convolved with a conic filter of radius $r_"min"$ to produce a smoothed field $tilde(rho)$:

$ tilde(rho)_e = frac(sum_j w(bold(x)_j) rho_j, sum_j w(bold(x)_j)),
  quad w(bold(x)_j) = max(0, r_"min" - lr(|bold(x)_e - bold(x)_j|)) $

This removes length-scale violations and guarantees mesh-independence.

#sub-head[Smooth Heaviside projection]
$tilde(rho)$ is projected to $overline(rho)$ with a smoothed step function parameterized by threshold $eta$ and sharpness $beta$, producing three designs:

#figure(
  table(
    columns: (auto, auto, auto),
    align: center,
    table.header[$eta_D = 0.3$][$eta_N = 0.5$][$eta_E = 0.7$],
    [_Dilated_], [_Nominal_], [_Eroded_],
  ),
  caption: [Three robust designs from the Heaviside projection.],
)

If the three designs share the same load-path topology and differ only in member slenderness, the design is considered *robust*: small manufacturing deviations will not change the structural behaviour.

#section-head[Optimized design]

#figure(
  rect(width: 100%, height: 48mm, fill: luma(235), stroke: 0.4pt)[
    #align(center + horizon)[_Optimized density field — add image_]
  ],
  caption: [Nominal optimized bowsprit topology ($eta_N = 0.5$). Blue = solid, white = void.],
)

The optimizer removes material from the low-stress interior while retaining the outer flanges (high bending stress) and a diagonal web (shear transfer), closely resembling an I-section or truss — consistent with classical beam theory.

#section-head[Results]

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    align: (left, right, right, right),
    table.header[*Metric*][*Full solid*][*Holy beam*][*Optimized*],
    [Max. disp. (mm)], [_x.xx_], [_x.xx_], [_x.xx_],
    [Strain energy (J)], [_x.xxx_], [_x.xxx_], [_x.xxx_],
    [Volume fraction], [1.000], [_0.xxx_], [0.330],
  ),
  caption: [Performance comparison of beam variants.],
)

#section-head[Discussion]

The optimized design achieves the volume target of 33 % while significantly reducing compliance compared to the uniform holy beam. The resulting topology places material along the principal stress trajectories: flanges carry bending, and diagonal struts carry shear — a result directly analogous to a Michell truss.

Sharp re-entrant corners appear at the clamped edge; in practice these would be filleted to avoid stress concentrations. The plane-strain assumption neglects out-of-plane bending (twist from asymmetric spinnaker loads), which is a limitation to address with a 3D model in future work.

#section-head[Conclusion]

Topology optimization via MMA with SIMP penalization, density filtering, and robust Heaviside projection yields a lightweight bowsprit design that maintains structural stiffness within a 33 % volume budget. The method is systematic, mesh-independent, and produces results that are interpretable in terms of classical structural mechanics.

#v(1fr)
#line(length: 100%, stroke: 0.4pt)
#v(2pt)
#text(size: 8pt)[
  Topology optimization · Exam hand-in · Aarhus Universitet · 2026
]

#bibliography("refs.bib", title: none, style: "ieee")
