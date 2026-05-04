#import "@preview/elsearticle:1.0.0": *
#import "@preview/mannot:0.3.1": *
#import "@preview/wordometer:0.1.5": word-count, word-count-of, total-words, total-characters

//#set page(numbering: "1 of 1")
//helper function for derivatives:
#let ded(upper, lower) = math.op($(partial #upper) / (partial #lower)$)

#show: elsearticle.with(
  title: "Topology optimization hand-in",
  authors: (
    (
      name: "Sigurd Mousten Jager Nielsen",
      affiliation: "Aarhus Universitet",
      corr: "202108107@uni.au.dk",
      id: none,
    ),
  ),
  journal: none,
  abstract: none,
  keywords: none,
  format: "preprint",
  // line-numbering: true,
)

//AU-LOGO
#figure(
  image("images/au_logo.png", width: 25%),
)
/*
#show: word-count.with(exclude: ())
#align(center)[In this hand-in there are #total-words words and #total-characters characters in sections @section_1 through @section_5]
*/
//PAGEBREAK
#colbreak()

#outline(title: "Table of contents")

#colbreak()

= Cantilevered beam without holes <section_1>
A 2D plane-strain linear-elastic cantilever beam was solved in FEniCS using the assignment geometry, material data and end load of 1 kN. The full beam gives a maximum displacement of 11.95 mm and a strain energy of 5.957 J. The implementation specifics can be seen in @strain_energy[section].

//The resulting displacement field can be seen here:

/*
#figure(
  image("images/wrench_result.png"),
   caption: "Spatial accelerations of the bodies 1 and 5 on the spaceshuttle"
)<wrench>
*/
= Discontinuous function rho_bar <section_2>
The holes are represented by a Discontinuous piecewise constant field $tilde(rho)$ set to $10^(-6)$ inside and $1$ outside the holes using a DG0 space. The implementation specifics can be seen in @piecewise[section] or in Topology.py.

= Variable youngs modulus <section_3>
Setting young's modulus to $E=E_0 dot tilde(rho)$ for the holy beam increases the maximum displacement value and strain energy from approximately 11.95 to 13.08 mm and 5.957 to 6.523 J respectively. This displacement increase is expected as material is removed due to the holes as seen in @rho_bar[figure].

= Area fraction of material <section_4>
The area fraction of material is calculated using: $A_"ratio" = frac(1,L dot H) integral.double_Omega tilde(rho) dif A$
The area fraction for the holy beam is approximately 0.805 and the implementation specifics can be seen in @material_area_function[section]. 

= Optimization problem using MMA <section_5>
To optimize the beam the strain energy is minimized subject to $A_"ratio" <= 0.8$ and first order forward finite differences were used to approximate the gradient.
The step size $Delta R$ of 2 mm was chosen to match the mesh size and avoid getting zero gradients when $Delta R << "mesh size"$.

#figure(
  align(center)[
  #table(
  align: (auto, left, left, left),
  columns: 4,
  table.header[metric\\name][*Full beam*][*Holy beam*][*Optimized beam*],
  [displacement],[11.95 mm],[13.08 mm],[12.63 mm],
  [strain energy],[5.957 J],[6.523 J],[6.295 J],
  [area fraction],[1.000],[0.805],[0.800],
)],
caption: [Comparison of beams]
)<comparison_table>
The optimized beam has a lower displacement than the holy beam while having a slightly lower area fraction as seen in @comparison_table[table].
#figure(
  grid(columns: 2,
  image("images/holy_rho_bar.png"),
  image("images/opt_rho_bar.png"),
),
caption: [The holy beam (left) and optimized beam (right)]
)<rho_bar>
The @rho_bar[figure] shows the optimized design reaching $A_"ratio" = 0.800$ and improves the holy beam from 13.08 mm to 12.63 mm in maximum displacement and from 6.523 J to 6.295 J in strain energy, while being lighter than the full beam. The increase in hole radii toward the free end is physically plausible, since the bending moment is largest near the fixed end and smaller closer to the applied load.

#colbreak()
= Code implementation <Code>
The hand-in is accompanied by three python files: mma.py (provided in course), beam_configurator.py and Topology_assignment.py that contain the code used for the hand-in.

Some relevant code snippets can be seen in the sections below:

== Strain energy: <strain_energy>

```python
    @property
def strain_energy(self):
    if self._displacement_field is not None:
        if self._strain_energy is None:
            t = self.geometry_properties.thickness
            u = self._displacement_field
            pot_energy = 0.5*t*fs.inner(self.stresses(u), self.strains(u))*fs.dx
            self._strain_energy = float(fs.assemble(pot_energy))
        return self._strain_energy
    else:
        raise RuntimeError("run solve() before calculating strain energy")
```

== Piecewise constant function: <piecewise>
```python
@property
def rho_bar(self)->fs.Function:
    if self._rho_bar is None:
        hole_function_space = fs.FunctionSpace(self.mesh, "DG", 0)
        self._rho_bar = fs.Function(hole_function_space, name="rho_bar")
        self._rho_bar.vector()[:] = 1.0 #set density fraction to 1 for all elements
    return self._rho_bar

```

== set_design function: <set_design>
```python
def set_design(self, radii_vector, near_zero_val=None):
    if near_zero_val is None:
        near_zero_val = self.near_zero_val
    
    radii = np.asarray(radii_vector, dtype=float).ravel() #ensure np.array 1D
    if radii.size !=10:
        raise ValueError(f"The radii vector should be 10 elements long, it was {radii.size} long")
    
    length = self.geometry_properties.length
    height = self.geometry_properties.height
    hole_centers = [(length/10.0*(hole+0.5), height/2.0) for hole in range(radii.size)] #find hole centers using provided formula in question 2.

    Vd = self.rho_bar.function_space()

    class BeamHoles(fs.UserExpression):
        def eval(self, value, x):
            val = 1.0
            for i, (center_x, center_y) in enumerate(hole_centers):
                if (x[0]-center_x)**2 + (x[1]-center_y)**2 <= radii[i]**2: #formula for circle
                    val = near_zero_val
                    break
            
            value[0] = val

        def value_shape(self):
            return ()

    temp_rho_bar = fs.interpolate(BeamHoles(element=Vd.ufl_element()), Vd).vector()
    self.rho_bar.vector()[:] = temp_rho_bar[:]

    self.reset_state(keep_displacement_field=True)

    return self.rho_bar
```

== material_area_fraction: <material_area_function>
```python
@property
def material_area_fraction(self): #this property is for question 4 in the hand-in
    length = self.geometry_properties.length
    height = self.geometry_properties.height

    return fs.assemble(self.rho_bar*fs.dx)/(length*height) #formula in q4
```

/* 
```python

```
*/

/*
#figure(
  image("images/wrench_result.png"),
   caption: "Spatial accelerations of the bodies 1 and 5 on the spaceshuttle"
)<wrench>
*/

#bibliography("refs.bib")