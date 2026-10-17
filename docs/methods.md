# Methods

## Diffusion

As mentioned in [Mechanics > Physics](./mechanics.md#physics) the world map is seen as a square periodic 2D grid of $N^2$ voxels.
Each voxel is a cube of side length $\Delta x$.
Molecule concentrations are modelled as well-mixed voxels in a [finite difference system](https://en.wikipedia.org/wiki/Finite_difference_method).
Flux from voxel $i$ to an adjacent voxel $j$ is

$$
J_{i \rightarrow j} = D \frac{c_i - c_j}{\Delta x}
$$

with a constant anisotropic [diffusion coefficient](https://en.wikipedia.org/wiki/Mass_diffusivity) $D$ and concentrations $c$.
For one voxel $(x, y)$ in the 2D grid its concentration changes with

$$
\frac{\text{d}c_{x,y}}{\text{d}t} =
\frac{D}{{\Delta x}^2} (c_{x+1,y} + c_{x-1,y} + c_{x,y+1} + c_{x,y-1} - 4 c_{x,y})
$$

with indices in modulo $N$.
This is the second order [finite difference Laplacian](https://en.wikipedia.org/wiki/Discrete_Laplace_operator) with period boundaries.
As matrix it is

$$
\frac{\text{d}\boldsymbol{C}}{\text{d}t} =
D \boldsymbol{\Delta}_N \boldsymbol{C}
\in \mathbb{R}^{N \times N}
$$

with $\boldsymbol{\Delta}_N$ being the periodic 2D graph Laplacian divided by ${\Delta x}^2$.
Its solution over arbitrary time intervals $\Delta t$ is

$$
\boldsymbol{C}(t + \Delta t) =
\boldsymbol{C}(t) e^{D \boldsymbol{\Delta}_N \Delta t}
$$

Because the finite-difference Laplacian is diagonalized by the 2D [discrete Fourier transform](https://en.wikipedia.org/wiki/Discrete_Fourier_transform) we can transform the concentration field, multiply each Fourier mode by its exact decay factor, then transform it back.
On a $N \times N$ periodic grid with spacing $\Delta x$, eigenvalues $\lambda$ for discrete Laplacian with Fourier indices $(m, n)$ are

$$
\lambda_{m,n} =
- \frac{4}{{\Delta x}^2} [\sin^2(\pi \frac{m}{N}) + \sin^2(\pi \frac{n}{N})] =
- \frac{1}{{\Delta x}^2} [4 - 2 \cos(2 \pi \frac{m}{N}) - 2 \cos(2 \pi \frac{n}{N})]
$$

Then, the update in the frequency domain is

$$
ĉ_{m,n}(t + \Delta t) = ĉ_{m,n}(t) e^{D \lambda_{m,n} \Delta t}
$$

The zero mode $m = n = 0$ has $\lambda = 0$ and so the total amount of molecules is conserved over time steps.
Additionally, the concentrations are guaranteed to not become negative.

## Permeation

As mentioned in [Mechanics > Physics](./mechanics.md#physics) a cell is approximated as a sphere swimming in a voxel of medium.
With [permeability coefficient](https://en.wikipedia.org/wiki/Permeability_(porous_media)) $P$ if the intracellular concentration of a molecule is $c_i$ and the extracellular concentration is $c_e$ then the flux
$J_{e \rightarrow i} = P (c_e - c_i)$
creates

$$
\frac{\text{d}c_i}{\text{d}t} = P \frac{A}{V_c} (c_e - c_i)
\text{ , }
\frac{\text{d}c_e}{\text{d}t} = -P \frac{A}{V_m} (c_e - c_i)
$$

with cell volume $V_c$, voxel volume $V_m$, and their interface area $A$.
Substracting both from one from one another shows the concentration difference decays with

$$
\text{d}(t + \Delta t) = \text{d}(t) e^{-k \Delta t}
\text{ , }
k = P A (\frac{1}{V_c} + \frac{1}{V_m}) 
$$

With no change of the total number of molecules during the permeation step, $c_i$ and $c_e$ will approach the equilibrium concentration

$$
c_{\infty} = \frac{V_c c_i + V_m c_e}{V_c + V_m}
$$

Thus, for an arbitrary time step each concentration can be calculated as

$$
c_i(t + \Delta t) = c_{\infty} + (c_i - c_{\infty}) e^{-k \Delta t} \\
c_e(t + \Delta t) = c_{\infty} + (c_e - c_{\infty}) e^{-k \Delta t}
$$


## Kinetics

[Mechanics > Kinetics](./mechanics.md#kinetics) explains how each cell can contain proteins which catalyze reactions.
The reaction velocity of a protein (or its catalyzed reaction) at any moment in time depends on the concentrations of involved molecules at that moment in time.
Since there can be multiple proteins in a cell with multiple molecule species a differential equation system is formed. Let  $\boldsymbol{N}$ be a matrix describing the reaction stoichiometry of all reactions catalyzed by proteins in a cell

$$
\boldsymbol{N} = \begin{pmatrix}
n_{1,1} & \cdots & n_{1,p} \\
\vdots  & \ddots & \vdots  \\
n_{m,1} & \cdots & n_{m,p} \end{pmatrix}
\in \mathbb{N}^{m \times p}
$$

where $n_{i,j}$ is the [stoichiometric number](https://en.wikipedia.org/wiki/Stoichiometry#Stoichiometric_coefficient_and_stoichiometric_number) of molecule species $i$ in reaction $j$ with $i \in 1 \cdots m$ for $m$ molecule species and $j \in 1 \cdots p$ for $p$ proteins (catalyzing reactions). The change of molecule concentrations over time at any point in time can then be described as ordinary differential equation (ODE) system

$$
\frac{\text{d}\boldsymbol{x}}{\text{d}t} = \boldsymbol{N} \boldsymbol{v}(\boldsymbol{x}(t))
$$

with molecule concentrations
$\boldsymbol{x}(t) = (x_1(t) \cdots x^{m)}(t))^\text{T} \in \mathbb{R}^m$
at time point $t$ for $m$ molecule species
and reaction velocities
$\boldsymbol{v} = (v_1(x) \cdots v_p(x))^\text{T} \in \mathbb{R}^p$
given concentrations $x$ for $p$ proteins catalyzing reactions.
This can be further simplified.
Over time molecule concentrations change according to $\boldsymbol{v}$ and their stoichiometric number in $\boldsymbol{N}$.
However, concentration changes for each reaction only really have one degree of freedom which can be described by [reaction extend](https://en.wikipedia.org/wiki/Extent_of_reaction) $\boldsymbol{\xi}$.

$$
\frac{\text{d} \boldsymbol{\xi}}{\text{d}t} =
\boldsymbol{v}(\boldsymbol{x}(t)) \Rightarrow \Delta \boldsymbol{\xi} =
\int_{t_n}^{t_{n+1}} \boldsymbol{v}(\boldsymbol{x}(t)) \text{d}t
$$

$\Delta \boldsymbol{\xi}$ is the change of molecule concentrations normalized by their stoichiometric number between time points
$t_n$ and $t_{n+1}$ with $t_{n+1} = t_n + \Delta t$.
When integrating the ODE $\text{d}\boldsymbol{x}/\text{d}t$ from above and substituting with $\Delta \boldsymbol{\xi}$ molecule concentrations $\boldsymbol{x_{n+1}}$ at time point $t_{n+1}$ can be described using molecule concentrations $\boldsymbol{x_n}$ at time point $t_n$

$$
\boldsymbol{x_{n+1}} = \boldsymbol{N} \Delta \boldsymbol{\xi} + \boldsymbol{x_n}
$$

With realistic reaction stoichiometry $\boldsymbol{v}(\boldsymbol{x})$ usually creates stiff ODEs.
However, within one time step $\boldsymbol{v}(\boldsymbol{x})$ is monotonously decreasing.
Thus, using the [backward Euler method](https://en.wikipedia.org/wiki/Backward_Euler_method) can prevent overshooting the fix point at $\boldsymbol{K_e}$ and possibly generating negative concentrations.

$$
\Delta \boldsymbol{\xi} = \int_{t_n}^{t_{n+1}} \boldsymbol{v}(\boldsymbol{x}(t)) \text{d}t \approx \Delta t \boldsymbol{v}(\boldsymbol{x_{n+t1}})
$$

Substituting $\boldsymbol{x_{n+1}}$ using $\Delta \boldsymbol{\xi}$ solve for $\Delta \boldsymbol{\xi^*}$ so that

$$
\Delta \boldsymbol{\xi^*} = \Delta t \boldsymbol{v}(\boldsymbol{N} \Delta \boldsymbol{\xi^*} + \boldsymbol{x_n})
$$

This ODE system is genuinely coupled since $\boldsymbol{N}$ can define reaction stoichiometries where multiple proteins use the same molecule species as substrates or products at the same time.
The solution can be approximated by using the [Jacobi method](https://en.wikipedia.org/wiki/Jacobi_method), splitting the $p$-dimensional problem into $p$ independent scalar problems.
Each scalar problem should have exactly one root which can be found using the [Bisection method](https://en.wikipedia.org/wiki/Bisection_method).
Consider $S$ block Jacobi sweeps with sweep index $k = 0 \cdots S - 1$.
First, calculate the background at step $k$.
For each protein $q \in 1 \cdots p$ calculate the background as

$$
\boldsymbol{b_q^{(k)}} = \boldsymbol{x_n} + \sum_{j \ne q}^p \boldsymbol{n_j} \Delta \xi_j^{(k)} \in \mathbb{R}^m
$$

where $\boldsymbol{n_j} \in \mathbb{R}^m$ is the reaction stoichiometry of protein $j$ with $\boldsymbol{N} = (\boldsymbol{n_1} \cdots \boldsymbol{n_p})$
and $\Delta \xi_j^{(k)}$ is the scalar $j$'th component of $\xi^{(k)}$.
Then, for each $q \in 1 \cdots p$ find $\Delta \xi_j^{*(k+1)}$ such that

$$
\Delta \xi_q^{*(k+1)} = \Delta t v_q (\boldsymbol{n_q} \Delta \xi_j^{*(k+1)} + \boldsymbol{b_q^{(k)}})
$$

where $\boldsymbol{n_q}$ are the reaction stoichiometry and $v_q$ the velocity function of protein $q$.
This equation should have exactly one root which can be found using the Bisection method. The search interval can be defined using the positivity requirement for concentrations. For each molecule $i$ the search interval for $\xi_q^{*(k+1)}$ is

$$
\max_{i: n_{i,q} > 0} \frac{-\boldsymbol{b_{i,q}^{(k)}}}{n_{i,q}}
\le
\xi_q^{*(k+1)}
\le
\min_{i: n_{i,q} < 0} \frac{\boldsymbol{b_{i,q}^{(k)}}}{-n_{i,q}}
$$

The whole approach can be summarized as the following:

For $k = 0 \cdots S - 1$:  
&nbsp;&nbsp;&nbsp;&nbsp;1. Compute background $\boldsymbol{B^{(k)}} = (\boldsymbol{b_1^{(k)}} \cdots \boldsymbol{b_p^{(k)}})$  
&nbsp;&nbsp;&nbsp;&nbsp;2. Compute boundaries for $\boldsymbol{\xi^{*(k+1)}} = (\xi_1^{*(k+1)} \cdots \xi_p^{*(k+1)})^\text{T}$  
&nbsp;&nbsp;&nbsp;&nbsp;3. Find $\boldsymbol{\xi^{*(k+1)}}$ using Bisection  
After $S$ sweeps return $\boldsymbol{x_{n+1}} = \boldsymbol{N} \Delta \boldsymbol{\xi^{*(S)}} + \boldsymbol{x_n}$