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
- what proteins a cell has and how exactly they work is defined by its genome
- genome is a string of ATCG
- genome is converted into coding regions, each coding region defines a protein
- each CDS can contain multiple domains
- domains can be catalytic, transporter, allosteric
- to what molecule they are specific and/or which reaction they catalyze is defined by genetic details of this domain
- how high the domains affinity to a certain molecule is and/or how quick it can catalyze a reaction is also defined by details
- each molecule species has an energetic state, an energy
- a reaction generally happens if it lowers the [Gibbs free energy](https://en.wikipedia.org/wiki/Gibbs_free_energy)
- so it depends on the current concentrations and the molecules and stoichiometry involved
- multiple domains on the same protein are energetically coupled
- so a reaction that increases Gibbs energy can be driven by another reaction that decreases it a bit more
- transporters can also drive a reaction and vice versa
- a reaction can make a transporter work against the concentration gradient
- additionally the activity of such catalytic domains and transporters can be regulated by additional allosteric domains
- all kinetics are [Michaelis-Menten-based](https://en.wikipedia.org/wiki/Michaelis%E2%80%93Menten_kinetics)
- allosteric regulation is [non-competitive](https://en.wikipedia.org/wiki/Non-competitive_inhibition)
- which reactions and molecule species exist is defined by the user
- after each step the user can interact with the world and its cells
- e.g. changing/mutating genomes, killing or replicating cells, changing molecule concetrations somewhere on the map
- thus one can create certain evolutionary pressures and mutation rates
- the simulation is implemented with millions of time steps in mind
- [PyTorch](https://pytorch.org/) was used as the main tool for fast computation and allowing calculations to be done on a GPU
- it is an ongoing effort to make this simualtion faster
- currently there are still some parts which have to be calculated on CPU, which are usually also the performance bottlenecks

### Genetics

All mechanisms are based on bacterial [transcription](<https://en.wikipedia.org/wiki/Transcription_(biology)>)
and [translation](<https://en.wikipedia.org/wiki/Translation_(biology)>).
A cell's [genome](https://en.wikipedia.org/wiki/Genome) is a chain of [nucleotides](https://en.wikipedia.org/wiki/Nucleotide) represented by a string of letters
$\{ T;C;G;A \}$. There are [start and stop codons](https://en.wikipedia.org/wiki/Genetic_code) which define [coding regions (CDSs)](https://en.wikipedia.org/wiki/Coding_region).
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
However, not if the ratio of products to substrates is too high ( $RT \ln Q$ ).

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
is defined in the domain itself as a region that encodes a boolean $\{0;1\}$.
All domains with orientation $0$ work from left to right, and _vice versa_ for $1$.
_E.g._ if there are 2 catalytic domains $A \leftrightharpoons B$ and $C \leftrightharpoons D$,
they would become $A + C \leftrightharpoons B + D$ if they have the same orientation,
and $A + D \leftrightharpoons B + C$ if not.

### Kinetics

### Implementation

```
python -m experiments.e0_performance.main --n_steps=10
...
tensorboard --logdir=./experiments/e0_performance/runs
```
