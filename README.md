## Magicsoup

This is a game that simulates cell metabolic and transduction pathway evolution.
Define a 2D world with certain molecules and reactions.
Add a few cells and create evolutionary pressure by selectively replicating and killing them.
Then run and see what random mutations can do.

![monitoring run](tensorboard_example.png "monitoring run")
_Watching an ongoing simulation using TensorBoard. In [this simulation](./experiments/e1_co2_fixing/) cells were made to fix CO2 from an artificial CO2 source in the middle of the map._

### Example

The basic building blocks of what a cell can do are defined by the world's chemistry.
There are molecules and reactions that can convert these molecules.
Cells can develop proteins with domains that can transport these molecules,
catalyze the reactions, and be regulated by molecules.
Any reaction or transport happens only if energetically favourable.
Below, I am defining the reaction $CO2 + NADPH \rightleftharpoons formiat + NADP$.
Each molecule species is defined with a fictional standard Gibbs free energy of formation.

```python
import torch
import magicsoup as ms

NADPH = ms.Molecule("NADPH", 200 * 1e3)
NADP = ms.Molecule("NADP", 100 * 1e3)
formiat = ms.Molecule("formiat", 20 * 1e3)
co2 = ms.Molecule("CO2", 10 * 1e3)

molecules = [NADPH, NADP, formiat, co2]
reactions = [([co2, NADPH], [formiat, NADP])]

chemistry = ms.Chemistry(reactions=reactions, molecules=molecules)
world = ms.World(chemistry=chemistry)
```

By coupling multiple domains within the same protein, energetically unfavourable actions
can be powered with the energy of energetically favourable ones.
These domains, their specifications, and how they are coupled in proteins, is all encoded in the cell's genome.
Here, I am generating 100 cells with random genomes of 500 basepairs each and place them
in random places on the 2D world map.

```python
genomes = [ms.random_genome(s=500) for _ in range(100)]
world.add_random_cells(genomes=genomes)
```

Cells discover new proteins by chance through mutations.
In the function below all cells experience 1E-3 random point mutations per nucleotide.
10% of them will be indels.

```python
def mutate_cells():
    mutated = ms.point_mutations(seqs=[d.genome for d in world.cells])
    world.update_cells(genome_idx_pairs=mutated)
```

Evolutionary pressure can be applied by selectively killing or replicating cells.
Here, cells have an increased chance of dying when formiat gets too low
and an increased chance of replicating when formiat gets high.

```python
def sample(p: torch.Tensor) -> list[int]:
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()

def kill_cells():
    x = world.cell_molecules[:, 2]
    idxs = sample(.01 / (.01 + x))
    world.kill_cells(cell_idxs=idxs)

def replicate_cells():
    x = world.cell_molecules[:, 2]
    idxs = sample(x ** 3 / (x ** 3 + 20.0 ** 3))
    world.replicate_cells(parent_idxs=idxs)
```

Finally, the simulation itself is run in a python loop by repetitively calling the different steps.
With `world.enzymatic_activity()` chemical reactions and molecule transport
in cells advance by one time step.
`world.diffuse_molecules()` lets molecules on the world map diffuse and permeate through cell membranes
(if they can) by one time step.

```python
for _ in range(1000):
    world.enzymatic_activity()
    kill_cells()
    replicate_cells()
    mutate_cells()
    world.diffuse_molecules()
    world.increment_cell_survival()
```

### Documentation

The user documentation is in the code.
All important classes and methods have docstrings explaining what they do,
how they work, and what they are used for.
For an explanation of the mechanics in this simulation please see [Concepts](#concepts) below.

In general, you create a `Chemistry` object with reactions and molecule species.
For molecules you can define things like energy, permeability, diffusivity.
See the [Molecule](./magicsoup/containers.py) docstring for more information.

Then, you create a `World` object which defines things like world map and genetics.
It carries all data that describes the world at this time step with cells, molecule distributions and so on.
On this object there are also methods that are used to advance the world by one time step.
Read the [World](./magicsoup/world.py) docstring for more information.

Usually, you would only adjust `Molecule`s and `World`.
However, in some cases you might want to change the way how genetics work,
_e.g._ change the way how certain domains are encoded, or change how coding regions are defined.
In that case you can override the default `Genetics` object.
See the [Genetics](./magicsoup/genetics.py) docstring for more information.

Apart from that, you create the simulation to your own likings.
From the `World` object you can read molecule abundances within cells and use that
to kill or replicate them (like in the example above).
You can also alter parts of the world, like creating concentration gradients
or regularly supplying the world with certain molecules/energy.

All major work is done by [PyTorch](https://pytorch.org/) and can be moved to a GPU.
The `World` object has an argument `device` to control that.
Please see [CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html) on how to use it.
And since this simulation already requires [PyTorch](https://pytorch.org/), it makes sense
to use [TensorBoard](https://pytorch.org/docs/stable/tensorboard.html) to interactively monitor your ongoing simulation.

## Concepts

The simulation is an agent-based 2D spatio-temporal simulation.
Cells are agents. Each cell has a string, the genome, which unambigously encodes a set or proteins, the proteome.
These proteins can change molecules in and around the cell.
Through that each cell can process information.
Each cell's proteome can form complex networks with feedback loops, cascades, and oscillators.
When randomly changing the cell's genome, this network changes randomly, too.
By selectively replicating certain cells while killing others, cells can be brought to evolve.

I believe that most interesting behaviors take many time steps to evolve.
That's why this simulation is implemented with millions of time steps in mind.
Genetics, chemistry, physics in this simulation is simplified a lot.
It is a tradeoff between a reasonable amount of complexity and performance.

- [Genetics](#genetics) explains how genetics work in this simulation
- [Chemistry](#chemistry) explanation for molecules, reactions and energy coupling
- [Kinetics](#kintics) explain the protein kinetics in this simulation
- [Implementation](#implementation) some implementation details worth mentioning

### Genetics

All mechanisms are based on bacterial [transcription](<https://en.wikipedia.org/wiki/Transcription_(biology)>)
and [translation](<https://en.wikipedia.org/wiki/Translation_(biology)>).
A cell's [genome](https://en.wikipedia.org/wiki/Genome) is a chain of [nucleotides](https://en.wikipedia.org/wiki/Nucleotide) represented by a string of letters
$\\{T;C;G;A\\}$. There are [start and stop codons](https://en.wikipedia.org/wiki/Genetic_code) which define [coding regions (CDSs)](https://en.wikipedia.org/wiki/Coding_region).
Transcription can start at any start codon and will end when the first [in-frame](https://en.wikipedia.org/wiki/Reading_frame) stop codon encountered.
CDSs without stop codon are not considered [[James NR 2016](https://pubmed.ncbi.nlm.nih.gov/27934701/)].
Each CDS will then be translated into one protein, giving each cell a certain [proteome](https://en.wikipedia.org/wiki/Proteome).
What each protein can do is defined by its [domains](https://en.wikipedia.org/wiki/Protein_domain).

Currently, there are three domain types: _catalytic, transporter, regulatory_.
Each domain consists of a region of genetic code that defines the domain type itself
and several regions that define its further specifications.
What these specifications are depends on the domain type.

_Catalytic_ domains can catalyze one of the user-defined reactions.
Specifications are the catalyzed reaction, affinities for substrates and products,
maximum velocity, its orientation (see [Energy](#energy)).

_Transporter_ domains can move a molecule species across the cell membrane,
_i.e._ from the outside world into the cell and _vice versa_.
Specifications are the molecule species, maximum velocity, its orientation (see [Energy](#energy)).

_Regulatory_ domains can regulate a protein through an effector molecule.
A protein with only a regulatory domain has no function.
But if the protein also has a catalytic or transporter domain, the regulatory
domain can up- or down-regulate this domain.
Specifications are the effector molecule species, whether it is an activating or inhibiting
effector, the affinity to that effector.

The exact genetic code for these domains is set when the `Genetics` object is instantiated.
But a user can also override the exact sequence-to-domain mappings.
This would be _e.g._ the exact sequence which will encode a catalytic domain for a certain reaction, with certain affinities and velocities.
In principle this flexibility allows a cell to create complex networks including feedback loops,
oscillators, and cascades.

For more details see [magicsoup/genetics.py](./magicsoup/genetics.py).
Also see [Kinetics](#kinetics) for details about the domain kinetics and aggregations.

### Chemistry

At the basis of this simulation one has to define which molecule species exist
and which reactions are possible.
On a high level, a more complex chemistry increases the search space
for a cell but also allows it to create more complex networks.

Every defined reaction can occur in both directions ($substrates \rightleftharpoons products$).
In which direction it will move for a particular cell and step is based on a mechanism
based on [Gibbs free energy](https://en.wikipedia.org/wiki/Gibbs_free_energy).
Each molecule species has an energy value, similar in principle to the
[Standard Gibbs free energy of formation](https://en.wikipedia.org/wiki/Standard_Gibbs_free_energy_of_formation).
Every reaction is regarded as the deconstruction of the substrates and the synthesis of the products.
During deconstruction the energy of all substrates is released, during synthesis the energy of all substrates is consumed.
This energy difference is defined as

$$\Delta G = \Delta G_0 + RT \ln Q$$

where $\Delta G_0$ is the standard Gibbs free energy of the reaction, $R$ is the [gas constant](https://en.wikipedia.org/wiki/Gas_constant), $T$ is the absolute temperature,
$Q$ is the [reaction quotient](https://en.wikipedia.org/wiki/Reaction_quotient).
The reaction that minimizes $\Delta G$ will occur.
So, generally the reaction that deconstructs high energy molecules and synthesizes low energy molecules will likely happen ( $\Delta G_0$ ).
However, it will turn around if the ratio of products to substrates is too high ( $RT \ln Q$ ).
There is an equilibrium $\Delta G_0 = RT \ln Q$ where the reaction stops.
When this equilibrium is approached the reaction or transporter will slow down and finally halt.

Each protein can have multiple domains and all domains of the same protein are energetically coupled.
So, an energetically unfavourable reaction can happen if at the same time another energetically
favourable reaction happens.
Transporter domains are also involved this way.
A transporter is seen as a reaction that converts an intracellular molecule species to its extracellular version (and _vice versa_).
Thus, for a transporter $\Delta G_0$ is always zero only the reaction quotient term $RT \ln Q$ is important.
A transporter can drive a reaction while molecules are allowed to diffuse along
their concentration gradient, or a reaction might drive a transporter to pump molecules
against their concentration gratient.

The sum of all $\Delta G$ of the protein domains dictates in which
direction the protein will work. In which orientation these domains are energetically coupled
is defined in the domain itself as a region that encodes a boolean $\\{0;1\\}$.
All domains with orientation $0$ work from left to right, and _vice versa_ for $1$.
_E.g._ if there are 2 catalytic domains $A \rightleftharpoons B$ and $C \rightleftharpoons D$,
they would become $A + C \rightleftharpoons B + D$ if they have the same orientation,
and $A + D \rightleftharpoons B + C$ if not.

For more details see [magicsoup/kinetics.py](./magicsoup/kinetics.py) where all the logic
for translating domains into kinetic parameters lives.
Also see [Implementation](#implementation) for some implications that arise from implementation details.

### Kinetics

All reactions in this simulation are based on [Michaelis-Menten-Kinetics](https://en.wikipedia.org/wiki/Michaelis%E2%80%93Menten_kinetics).
If a cell has one protein with one _catalytic domain_ that defines $S \rightleftharpoons P$ it will create molecule species $P$ from $S$ with a rate of

$$v = V_{max} \frac{[S]}{[S] + K_m} = \Delta [P] = -\Delta [S]$$

where $V_{max}$ is the maximum velocity of that reaction, $[S]$ is the amount of substrate available,
$K_m$ is the Michaelis constant.
When a reaction involves multiple substrate species and/or multiple catalytic domains are
aggregated this becomes

$$v = V_{max} \prod_{i} \frac{[S_i]^{n_i}}{([S_i] + K_{mi})^{n_i}}$$

where $[S_i]$ is the amount of substrate $i$ available, $n_i$ is the [stoichiometric coefficient](https://en.wikipedia.org/wiki/Chemical_equation#Structure) of substrate $i$, $K_{mi}$ is the Michaelis constant for substrate $i$.
What exactly these values are is encoded in the domain itself (see [Genetics](#genetics)).

_Transporter domains_ essentially work in the same way. They are defined as $[A_{int}] \rightleftharpoons [A_{ext}]$ where $A_{int}$ is a molecule species $A$ inside the cell and $A_{ext}$ is the same molecule species outside the cell.

_Regulator domains_ are also described by the same kinetic. However, they don't have $V_{max}$.
Their activity is defined by

$$a = \frac{[E]}{[E] + K_{mE}}$$

where $[E]$ is the amount of effector molecule available, $K_{mE}$ is the Michaelis constant for that effector molecule.
As with multiple substrates, multiple regulatory domains are combined over a product.
Depending on whether they are activating or inhibiting, they will be multiplied with $v$ in a different way thus creating a [non-competitive regulation](https://en.wikipedia.org/wiki/Non-competitive_inhibition).
A protein with regulatory domains will have a regulated velocity of

$$v = a_a (1 - a_i)V_{max} \prod_{i} \frac{[S_i]^{n_i}}{([S_i] + K_{mi})^{n_i}}$$

where $a_a$ is the combined activity of all activating effectors and $a_i$ is the combined activity of all inhibiting effectors.
As effector activities are $a \in [0;1)$ a regulatory effector cannot increase the maximum
velocity of a protein.
Also note that while an unregulated protein can always be active, a protein with an activating
regulatory domain can only be active if the activating effector is present.
So, an activating regulatory domain can also switch off a protein.

### Implementation

Making the simulation more performant is an oingoing effort.
I want to get the frequency up to a thousand time steps per second for a simulation with around 100k cells.
Currently, the simulation leans on [PyTorch](https://pytorch.org/) to do that.
Most operations are expressed as pytorch operations and can be moved to a GPU.
However, there are still parts which are calculated in plain python.
As of now, these are the operations concerned with creating/mutating genomes, transcription and translation.
These parts are usually the performance bottlenecks.

#### Low molecule abundances

Changes in molecule abundances are calculated for every step based on protein velocities.
These protein velocities depend in one part on substrate abundances
(Michaelis-Menten Kinetics as desribed in [Kinetics](#kinetics)).
Thus, generally as substrate abundances decrease, protein velocities decrease.
And so, deconstruction rates of this molecule species decrease.
Furthermore, as the ratio of products to substrates gets too high, the reaction stops or turns around
(free Gibbs energy of the reaction as described in [Energy](#energy)).
So, generally proteins shouldn't attempt to deconstruct more substrate than possible.
However, if a protein has a very high $V_{max}$ and a very low $K_M$ it can happen that
during one step it would deconstruct more substrate than actually available.
This can also happen if multiple proteins in the same cell would deconstruct the same molecule species.

To avoid deconstructing more substrate than available and creating negative molecule abundances there is a safety mechanism.
First, the naive protein velocities $v$ are calculated and compared with substrate abundances.
If some $v$ attempts to deconstruct more substrate than available, it is reduced to
by a factor to leave almost zero substrates (a small constant $\varepsilon$ is kept).
All protein velocities in the same cell are reduced by the same factor.
This is because of possible dependencies between proteins.
Say, protein P0 tried to do $A \rightleftharpoons B$ with $v_{P0} = 2$, but only 1 of A was available.
At the same time another protein P1 in the same cell does $B \rightleftharpoons C$ with $v_{P1} = 2$, with 0.5 of B available.
In the naive calculation P1 would be valid because P0 would create 2 B and so P1 can legitimately deconstruct 2 B.
However, after the naive calculation P0 is slowed down with a factor of almost 0.5, which means
it now deconstructs almost 1 A and synthesizes almost 1 B.
Now, P1 became a downstream problem of reducing P0, as it doesn't have enough B.
To avoid calculating a dependency tree during each step for each cell, all proteins are slowed down by the same factor.

Doing a lazy limit (_e.g._ `X.clamp(0.0)`) is also not an option.
This would mean a cell could deconstruct only 1 A while gaining 2 B.
It would create molecules and energy from nothing.
This sounds like an unlikely event, but the cells will exploit this (personal experience).
See `Kinetics.integrate_signals` in [magicsoup/kinetics.py](./magicsoup/kinetics.py) for more information.

#### Integrating multiple domains

I had to make a decision with $V_{Max}$ and $K_M$ when having proteins with multiple domains.
When there are multiple _e.g._ catalytic domains, it might make sense to each give them a seperate
$V_{Max}$. But then I would need to consider that different domains within the same protein
work at different rates. Thus, the whole energy coupling would become more tricky. _E.g._ should the protein be allowed to do 10x reaction 1 with $\Delta G = -1.1$ to power 1x reaction 2 with $\Delta G = 10$? To avoid such problems I decided to give any protein only a single $V_{Max}$. All $V_{Max}$ that might come from multiple domains are averaged to a single value. That means proteins with many domains tend to have less extreme values for $V_{Max}$.

A similar problem arises with $K_M$: multiple domains can attempt to each give a different $K_M$ value to the same molecule species. _E.g._ there could be a catalytic domain that has molecule A as a substrate and a regulatory domain with molecule A as effector. In these cases I decided to also only have 1 value for $K_M$ for each molecule species. All $K_M$ values for the same molecule species in the same protein are averaged.

#### Energetic Equilibrium

Theoretically, a reaction should occur in one direction according to its free Gibbs energy $\Delta G$. At some point $\Delta G = 0$ is approached
and the reaction should be in an equilibrium state where no appreciable difference in substrates and products is measurable anymore.
As transporters in this simulation also function like catalytic domains, the below is also true for transporters.

All reaction quotients are compared with their equilibrium constants and turned around if energetically unfavourable.
Then, quotients and equilibrium constants are used again to calculate a factor for selectively slowing down proteins close to or at their equilibrium.
This factor is multiplied with the final protein velocity.
It is calculated with an arbitrary function:

$$
f(x) =
\begin{cases}
  1 & if \quad |x| >= 1 \\
  |x| & if \quad  0.1 < |x| < 1 \\
  0 & otherwise
\end{cases}
\quad , \quad
x = \ln Q - \ln K_E = \ln Q + \frac{E}{RT}
$$

However, proteins with large $V_{max}$ and low $K_M$ can overshoot this equilibrium step.
In that case $\ln Q \gg \ln K_E$ in one step, and $\ln Q \ll \ln K_E$ in the next.
To avoid endless jumping back and forth around the equilibrium state $K_M$ of some domains are directional.
During translation the value of $K_M$ for a specific domain is read from the nucleotide sequence.
For catalytic domains and transporter domains this value of $K_M$ is set for its substrate
and its reciprocal $K_M^{-1}$ is set for its product.
This means if a protein was very sensitive to a substrate and overshot the equilibrium state,
it will be very unsensitive to the product (which will then be the substrate).
Thus, the protein might quickly approach and overshoot the equilibrium state from one side,
but then slowly approach it from the other (and hopefully reach the $|x| < 1$ interval).

One implication or observation from this is, that it is not good to have huge values for $V_{max}$.
The range of values to draw $V_{max}$ from should not have values much higher than 10 per time step.
If one wants to simulate much higher protein velocities (such as a catalase)
it would be better to just call `world.enzymatic_activity()` multiples times.
