# Reaction–Diffusion Pattern Formation

This project looks at how spatial patterns (Turing patterns) emerge in reaction–diffusion systems through a combination of linear stability analysis and numerical simulation.

The goal is to understand how simple local reaction rules, when combined with diffusion, can produce complex spatial structure. In particular, we study how a homogeneous steady state becomes unstable and reorganises into patterns such as spots, stripes, and more irregular structures.

All simulations are performed on **rectangular domains with Neumann (zero-flux) boundary conditions**, meaning the admissible spatial modes are cosine modes of the form

$$
\cos\left(\frac{n\pi x}{L_x}\right)\cos\left(\frac{m\pi y}{L_y}\right),
$$

Thus, only discrete spatial modes are admissible, and pattern formation is constrained by the domain geometry. This plays a central role in determining which patterns can form in practice.

---

## Example

<p align="center">
  <img src="outputs/gifs/schnakenberg_2D_both.gif" width="600">
</p>
<p align="center"><i>Evolution of U and V concentrations in the Schnakenberg system</i></p>

This animation shows the evolution of the Schnakenberg system from a small perturbation into a spatially structured steady state.

---

## Models implemented

### Schnakenberg model
\[
f(u,v) = a - u + u^2 v, \quad g(u,v) = b - u^2 v
\]

- Closed-form steady state
- Standard benchmark for Turing instability

---

### Substrate–Inhibitor model
\[
f(u,v) = a - u - \frac{\rho u v}{1 + u + K u^2}, \quad
g(u,v) = \alpha(b - v) - \frac{\rho u v}{1 + u + K u^2}
\]

- Nonlinear saturating inhibition
- Strong parameter sensitivity
- Steady state computed numerically

---

### Gierer–Meinhardt model
\[
f(u,v) = a - b u + \frac{u^2}{v}, \quad
g(u,v) = u^2 - v
\]

- Strong activator–inhibitor feedback
- Produces sharp localised structures

---

## Numerical method (AB2–AM2 scheme)

All simulations use a second-order semi-implicit scheme:

- **Adams–Bashforth (AB2)** for nonlinear reaction terms (explicit)
- **Adams–Moulton (AM2)** for diffusion (implicit)

The scheme is:

\[
U^{n+1} = M^{-1} \left( P U^n + \frac{3}{2}F(U^n) - \frac{1}{2}F(U^{n-1}) \right)
\]

where:
- \( M = I - \frac{\Delta t}{2}L \)
- \( P = I + \frac{\Delta t}{2}L \)
- \( L \) is the discrete Laplacian
- \( F(U) \) represents reaction terms

This provides:
- stability for stiff diffusion terms  
- second-order accuracy in time  
- efficient sparse linear solves  

---

## What this project demonstrates

- Linear stability and **Turing conditions**
- Dispersion relation and unstable wavenumber bands
- Mode selection in bounded domains
- Nonlinear saturation of patterns
- Dependence on:
  - diffusion ratio \( d \)
  - domain size \( (L_x, L_y) \)
  - aspect ratio
  - initial perturbation

---

## Repository structure

```
src/
  models/
    schnakenberg.py
    substrate_inhibitor.py
    gierer_meinhardt.py

  solvers/
    solver_1D.py
    solver_2D.py

  visualiser/
    plot_1D.py
    plot_2D.py
    animation_2D.py
    phase_plane.py
    turing_relation.py

  analysis_tools/
    derivatives.py
    steady_states.py
    turing.py

notebooks/
  analysis/
    schnakenberg.ipynb
    substrate_inhibitor.ipynb
    gierer_meinhardt.ipynb

  demos/
    schnakenberg_1D.ipynb
    schnakenberg_2D.ipynb
    substrate_inhibitor_2D.ipynb
    gierer_meinhardt_2D.ipynb

outputs/
  gifs/
  images/

LICENSE
README.md
```

---

## Running a simulation

Example (2D Schnakenberg):

```python
from src.solvers import AB2AM2Solver2D
from src.models import SchnakenbergModel

params = {
    "a": 0.1,
    "b": 0.9,
    "gamma": 2.0,
    "d": 20.0
}

solver = AB2AM2Solver2D(
    model=SchnakenbergModel,
    params=params,
    Lx=100,
    Ly=100,
    Nx=100,
    Ny=100,
    T=100,
    dt=0.1
)

t, U, V, x, y, X, Y = solver.run(p_type="random")
```

---

## Key observations

- Only a **band of spatial modes** becomes unstable, as predicted by the dispersion relation  
- Domain geometry restricts admissible modes, so the observed pattern depends strongly on \(L_x, L_y\) and aspect ratio  
- The final pattern is determined by:
  - fastest-growing modes during the linear instability stage  
  - nonlinear interactions and saturation in the fully developed regime  
- The Schnakenberg model produces stable spot and stripe regimes, while the substrate–inhibitor and Gierer–Meinhardt systems exhibit greater sensitivity and more fragmented or localised structures  

---

## License

This project is licensed under the MIT License (see `LICENSE` file for details).

---

## Author

Chamu V
