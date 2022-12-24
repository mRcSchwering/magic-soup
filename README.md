## Magicsoup

This is a game that simulates cell metabolic and transduction pathway evolution.
Define a 2D world in which certain molecules and reactions are possible.
Add some cells in and create some evolutionary pressure.
Then see what random mutations can create over time.

### Concepts

This game is a 2D spatio-temporal simulation.
The world map of this game is a square 2D grid on which cells and molecules can exist.
Time is a step-by-step calculation of updates for each pixel and cell in this world.

- molecule species diffuse freely over map
- molecules degrade
- cell is a compartment and has its own intracellular molecules
- cell is in interaction with the molecules of the pixel it occupies
- cell can change molecules by catalyzing reactions
- cell can transport molecules into or out
- cell proteins can be regulated by allosteric effectors
- ...
- thus one can create certain evolutionary pressures and mutation rates

### Genetics

All mechanisms are based on bacterial [transcription](<https://en.wikipedia.org/wiki/Transcription_(biology)>)
and [translation](<https://en.wikipedia.org/wiki/Translation_(biology)>).
A cell's [genome](https://en.wikipedia.org/wiki/Genome) is a chain of [nucleotides](https://en.wikipedia.org/wiki/Nucleotide) represented by a string of letters
$\\{T;C;G;A\\}$. There are [start and stop codons](https://en.wikipedia.org/wiki/Genetic_code) which define [coding regions (CDSs)](https://en.wikipedia.org/wiki/Coding_region).
Transcription can start at any start codon and will end when the first [in-frame](https://en.wikipedia.org/wiki/Reading_frame) stop codon encountered.
CDSs without stop codon are not considered [[James NR 2016](https://pubmed.ncbi.nlm.nih.gov/27934701/)].
Each CDS will then be translated into one protein, giving each cell a certain [proteome](https://en.wikipedia.org/wiki/Proteome).
What each protein can do is defined by its [domains](https://en.wikipedia.org/wiki/Protein_domain).

Currently, there are three domain types: _catalytic, transporter, allosteric_.
Each domain consists of a region of genetic code that defines the domain type itself
and several regions that define its details.
What these details are depends on the domain type.

_Catalytic_ domains can catalyze one of the user-defined reactions.
Details are the catalyzed reaction, affinities for substrates and products,
maximum velocity, its orientation (see [Energy](#energy)).

_Transporter_ domains can move a molecule species across the cell membrane,
_i.e._ from the outside world into the cell and _vice versa_.
Details are the molecule species, maximum velocity, its orientation (see [Energy](#energy)).

_Allosteric_ domains can regulare a protein through an effector molecule.
A protein with only an allosteric domain has no function.
But if the protein also has a catalytic or transporter domain, the allosteric
domain can up- or down-regulate this domain.
Details are the effector molecule species, whether it is an activating or inhibiting
effector, the affinity to that effector.

The exact genetic code for these domains is defined by the user.
This would be _e.g._ the exact sequence which will encode a catalytic domain for a certain reaction, with certain affinities and velocities.
As it makes sense to define a multitude of these domain definitions, there are factories
that help with their creation.

For more details see [magicsoup/genetics.py](./magicsoup/genetics.py).
Also see [Kinetics](#kinetics) for details about the domain kinetics and aggregations.

### Energy

Every defined reaction can occur in both directions ($substrates \leftrightharpoons products$).
In which direction it will occur at a particular time step in a particular cell depends
on a mechanism based on [Gibbs free energy](https://en.wikipedia.org/wiki/Gibbs_free_energy).
Each molecule species has an energy value.
Every reaction from a substrate to a product is regarded as the deconstruction of the substrates
and the synthesis of the products. During deconstruction the energy of all substrates is released, during synthesis the energy of all substrates is consumed.
This energy difference is defined as

$$\Delta G = \Delta G_0 + RT \ln Q$$

where $\Delta G_0$ is the standard Gibbs free energy of the reaction, $R$ is the [gas constant](https://en.wikipedia.org/wiki/Gas_constant), $T$ is the absolute temperature,
$Q$ is the [reaction quotient](https://en.wikipedia.org/wiki/Reaction_quotient).
The reaction that minimizes $\Delta G$ will occur.
So, generally the reaction that deconstructs high energy molecules and creates low energy molecules will likely happen ( $\Delta G_0$ ).
However, it will turn around if the ratio of products to substrates is too high ( $RT \ln Q$ ).
There is an equilibrium state were $\Delta G = 0$ and no reaction happens.

Each protein can have multiple domains and all domains of the same protein are energetically coupled.
So, an energetically unfavourable reaction can happen if at the same time another energetically
favourable reaction happens.
Transporter domains are also involved this way.
For transporter domains only the entropy term $RT \ln Q$ is important.
Thus, a transporter can drive a reaction while molecules are allowed to diffuse along
their concentration gradient, or a reaction might drive a transporter to pump molecules
against their concentration gratient.

The sum of all $\Delta G$ of the protein domains dictates in which
direction the protein will work. In which orientation these domains are energetically coupled
is defined in the domain itself as a region that encodes a boolean $\\{0;1\\}$.
All domains with orientation $0$ work from left to right, and _vice versa_ for $1$.
_E.g._ if there are 2 catalytic domains $A \leftrightharpoons B$ and $C \leftrightharpoons D$,
they would become $A + C \leftrightharpoons B + D$ if they have the same orientation,
and $A + D \leftrightharpoons B + C$ if not.

For more details see [magicsoup/kinetics.py](./magicsoup/kinetics.py) where all the logic
for translating domains into kinetic parameters lives.
Also see [Implementation](#implementation) for some implications that arise from implementation details.

### Kinetics

All reactions in this simulation are based on [Michaelis-Menten-Kinetics](https://en.wikipedia.org/wiki/Michaelis%E2%80%93Menten_kinetics). If a cell has one protein with one catalytic domain that defines $S \leftrightharpoons P$ it will create molecule species $P$ from $S$ with a rate of

$$v = V_{max} \frac{[S]}{[S] + K_m} = \Delta [P] = -\Delta [S]$$

where $V_{max}$ is the maximum velocity of that reaction, $[S]$ is the amount of substrate available,
$K_m$ is the Michaelis constant.
When a reaction involves multiple substrate species and/or multiple catalytic domains are
aggregated this becomes

$$v = V_{max} \prod_{i} \frac{[S_i]^{n_i}}{([S_i] + K_{mi})^{n_i}}$$

where $[S_i]$ is the amount of substrate $i$ available, $n_i$ is the [stoichiometric coefficient](https://en.wikipedia.org/wiki/Chemical_equation#Structure) of substrate $i$, $K_{mi}$ is the Michaelis constant for substrate $i$.
What exactly these values are is encoded in the domain itself (see [Genetics](#genetics)).

Transporters essentially work in the same way. They are defined as $[A_{int}] \leftrightharpoons [A_{ext}]$ where $A_{int}$ is a molecule species $A$ inside the cell and $A_{ext}$ is the same molecule species outside the cell.

Allosteric domains are also defined by the same kinetic. However, they don't have $V_{max}$.
Their activity is defined by

$$a = \frac{[E]}{[E] + K_{mE}}$$

where $[E]$ is the amount of effector molecule available, $K_{mE}$ is the Michaelis constant for that effector molecule.
As with multiple substrates, multiple allosteric domains are combined over a product.
Depending on whether they are activating or inhibiting, they will be multiplied with $v$ in a different way thus creating a [non-competitive regulation](https://en.wikipedia.org/wiki/Non-competitive_inhibition).
A protein with allosteric domains will have a regulated velocity of

$$v = a_a (1 - a_i)V_{max} \prod_{i} \frac{[S_i]^{n_i}}{([S_i] + K_{mi})^{n_i}}$$

where $a_a$ is the combined activity of all activating effectors and $a_i$ is the combined activity of all inhibiting effectors.
As effector activities are $a \in [0;1)$ an allosteric effector cannot increase the maximum
velocity of a protein.
Also note that while an unregulated protein can always be active, a protein with an activating
allosteric domain can only be active if the activating effector is present.
So, an activating allosteric domain can also switch off a protein.

### Implementation

- the simulation is implemented with millions of time steps in mind
- [PyTorch](https://pytorch.org/) was used as the main tool for fast computation and allowing calculations to be done on a GPU
- it is an ongoing effort to make this simualtion faster
- currently there are still some parts which have to be calculated on CPU, which are usually also the performance bottlenecks

```
python -m experiments.e0_performance.main --n_steps=10
...
tensorboard --logdir=./experiments/e0_performance/runs
```
