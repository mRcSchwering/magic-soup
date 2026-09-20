# Computation

Simulation mechanics are explained in [mechanics](./mechanics.md).
Each cell can contain proteins which catalyze reactions.
As the **Kinetics** section explains the reaction velocity of a protein (or its catalyzed reaction) at any moment in time depends on the concentrations of involved molecules at that moment in time.
Since there can be multiple proteins in a cell molecule concentrations form a differential equation system which has to be solved in order to calculate molecule concentrations at any moment in time.

## Problem

Let  $\boldsymbol{N}$ be a matrix describing the reaction stoichiometry of all reactions catalyzed by proteins in a cell

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
\frac{\text{d} \boldsymbol{\xi}}{\text{d}t} = \boldsymbol{v}(\boldsymbol{x}(t)) \Rightarrow \Delta \boldsymbol{\xi} = \int_{t_n}^{t_{n+1}} \boldsymbol{v}(\boldsymbol{x}(t)) \text{d}t
$$

$\Delta \boldsymbol{\xi}$ is the change of molecule concentrations normalized by their stoichiometric number between time points
$t_n$ and $t_{n+1}$ with $t_{n+1} = t_n + h$.
When integrating the ODE $\text{d}\boldsymbol{x}/\text{d}t$ from above and substituting $\Delta \boldsymbol{\xi}$ molecule concentrations can be described as

$$
\boldsymbol{x_{n+1}} = \boldsymbol{N} \Delta \boldsymbol{\xi} + \boldsymbol{x_n}
$$

with molecule concentrations $\boldsymbol{x_n}$ at time point $t_n$ and molecule concentrations $\boldsymbol{x_{n+1}}$ at time point $t_{n + 1}$.

## Approximation

In realistic simulations $\boldsymbol{v}(\boldsymbol{x})$ usually creates stiff ODEs.
However, within one time step $\boldsymbol{v}(\boldsymbol{x})$ is monotonously decreasing.
Thus, using the [backward Euler method](https://en.wikipedia.org/wiki/Backward_Euler_method) should prevent overshooting the fix point at $\boldsymbol{K_e}$ (see [mechanics](./mechanics.md)) and possibly generating negative concentrations.

$$
\Delta \boldsymbol{\xi} = \int_{t_n}^{t_{n+1}} \boldsymbol{v}(\boldsymbol{x}(t)) \text{d}t \approx h \boldsymbol{v}(\boldsymbol{x_{n+t1}})
$$

Substituting $\boldsymbol{x_{n+1}}$ using $\Delta \boldsymbol{\xi}$ solve for $\Delta \boldsymbol{\xi^*}$ so that

$$
\Delta \boldsymbol{\xi^*} = h \boldsymbol{v}(\boldsymbol{N} \Delta \boldsymbol{\xi^*} + \boldsymbol{x_n})
$$

This is a genuinely coupled ODE system since $\boldsymbol{N}$ can define reaction stoichiometries where multiple proteins use the same molecule species as substrates or products at the same time.
The system can be approximated by using the [Jacobi method](https://en.wikipedia.org/wiki/Jacobi_method), splitting the $p$-dimensional problem into $p$ independent scalar problems.
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
\Delta \xi_q^{*(k+1)} = h v_q (\boldsymbol{n_q} \Delta \xi_j^{*(k+1)} + \boldsymbol{b_q^{(k)}})
$$

where $\boldsymbol{n_q}$ are the reaction stoichiometry and $v_q$ the velocity function of protein $q$.
This equation should have exactly one root which can be found using the Bisection method. The search interval can be defined using the positivity requirement for concentrations. For each molecule $i$ the search interval for $\xi_q^{*(k+1)}$ is

$$
\max_{i: n_{i,q} > 0} \frac{-b_{i,q}^{(k)}}{n_{i,q}}
\ge
\xi_q^{*(k+1)}
\ge
\min_{i: n_{i,q} < 0} \frac{b_{i,q}^{(k)}}{-n_{i,q}}
$$

The whole approach can be summarized as the following:

For $k = 0 \cdots S - 1$:  
&nbsp;&nbsp;&nbsp;&nbsp;1. Compute background $\boldsymbol{B^{(k)}} = (\boldsymbol{b_1^{(k)}} \cdots \boldsymbol{b_p^{(k)}})$  
&nbsp;&nbsp;&nbsp;&nbsp;2. Compute boundaries for $\boldsymbol{\xi^{*(k+1)}} = (\xi_1^{*(k+1)} \cdots \xi_p^{*(k+1)})^\text{T}$  
&nbsp;&nbsp;&nbsp;&nbsp;3. Find $\boldsymbol{\xi^{*(k+1)}}$ using Bisection  
After $S$ sweeps return $\boldsymbol{x_{n+1}} = \boldsymbol{N} \Delta \boldsymbol{\xi^{*(S)}} + \boldsymbol{x_n}$