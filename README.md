## Magicsoup

This is a game that simulates cell metabolic and transduction pathway evolution.
Define a 2D world in which certain molecules and reactions are possible.
Add a few cells and create some evolutionary pressure.
Then run and see what random mutations can create over time.

### Example

The complexity of each cell is essentially limited by
the molecules and protein domains you define.
These domains can be combined and energetically coupled in proteins in any number and orientation.

```python
NADPH = Molecule("NADPH", 200.0)
NADP = Molecule("NADP", 100.0)
formiat = Molecule("formiat", 20.0)
co2 = Molecule("CO2", 10.0)

molecules = [NADPH, NADP, formiat, co2]
reactions = [([co2, NADPH], [formiat, NADP])]

reaction_map = generic_map_fact("NNNNNN", reactions)
molecule_map = generic_map_fact("NNNNNN", molecules)
affinity_map = weight_map_fact("NNNNNN", 0.1, 10.0)
velocity_map = weight_map_fact("NNNNNN", 1.0, 10.0)
bool_map = bool_map_fact("NNNNNN")

domains = {
    ms.CatalyticFact(reaction_map=reaction_map, affinity_map=affinity_map, velocity_map=velocity_map, orientation_map=bool_map):
        ms.variants("ACNTGN") + ms.variants("AGNTGN") + ms.variants("CCNTTN"),
    ms.TransporterFact(molecule_map=molecule_map, affinity_map=affinity_map, velocity_map=velocity_map, orientation_map=bool_map):
        ms.variants("ACNAGN") + ms.variants("ACNTAN") + ms.variants("AANTCN"),
}

world = ms.World(domain_facts=domains, molecules=molecules)
world.summary()
```

Cells discover proteins by chance through transcription and translation of their genome.
To promote odds of this happening you might want to add new cells with new random genomes to the world.
Or you mutate genomes of already existing cells.

```python
def add_random_cells():
    genomes = [ms.random_genome(s=100) for _ in range(100)]
    world.add_random_cells(genomes=genomes)

def mutate_cells():
    gs, idxs = ms.point_mutations(seqs=[d.genome for d in world.cells])
    world.update_cells(genomes=gs, idxs=idxs)
```

To create evolutionary pressure we can decide to kill certain cells,
and let certain other cells replicate.
In the example below, the condition for each is simply based on the intracellular concentrations
of a molecule species.

```python
def kill_cells(mol_idx: int):
    idxs = (
        torch.argwhere(world.cell_molecules[:, mol_idx] < 1.0)
        .flatten()
        .tolist()
    )
    world.kill_cells(cell_idxs=idxs)

def replicate_cells(mol_idx: int):
    idxs = (
        torch.argwhere(world.cell_molecules[:, mol_idx] > 5.0)
        .flatten()
        .tolist()
    )
    succ_parents, children = world.replicate_cells(parent_idxs=idxs)
    world.cell_molecules[succ_parents + children, mol_idx] -= 4.0
```

To actually start the simulation we repetitively apply `world.enzymatic_activity()`
and run our functions to generate new genomes and apply evolutionary pressure.

```python
idx_NADPH = NADPH.idx

for step_i in range(1000):
    add_random_cells()
    world.enzymatic_activity()  # catalyze reactions
    kill_cells(idx_NADPH)
    replicate_cells(idx_NADPH)
    mutate_cells()
    world.diffuse_molecules()
    world.increment_cell_survival()
```

## Details

- [Genetics](#genetics) explains how genetics work in this simulation
- [Energy](#energy) explanation for molecule energies and energy coupling
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

Every defined reaction can occur in both directions ($substrates \rightleftharpoons products$).
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
So, generally the reaction that deconstructs high energy molecules and synthesizes low energy molecules will likely happen ( $\Delta G_0$ ).
However, it will turn around if the ratio of products to substrates is too high ( $RT \ln Q$ ).
An equilibrium state can be reached where $\Delta G = 0$ and no reaction happens (# TODO: realy?).

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
_E.g._ if there are 2 catalytic domains $A \rightleftharpoons B$ and $C \rightleftharpoons D$,
they would become $A + C \rightleftharpoons B + D$ if they have the same orientation,
and $A + D \rightleftharpoons B + C$ if not.

For more details see [magicsoup/kinetics.py](./magicsoup/kinetics.py) where all the logic
for translating domains into kinetic parameters lives.
Also see [Implementation](#implementation) for some implications that arise from implementation details.

### Kinetics

All reactions in this simulation are based on [Michaelis-Menten-Kinetics](https://en.wikipedia.org/wiki/Michaelis%E2%80%93Menten_kinetics). If a cell has one protein with one _catalytic domain_ that defines $S \rightleftharpoons P$ it will create molecule species $P$ from $S$ with a rate of

$$v = V_{max} \frac{[S]}{[S] + K_m} = \Delta [P] = -\Delta [S]$$

where $V_{max}$ is the maximum velocity of that reaction, $[S]$ is the amount of substrate available,
$K_m$ is the Michaelis constant.
When a reaction involves multiple substrate species and/or multiple catalytic domains are
aggregated this becomes

$$v = V_{max} \prod_{i} \frac{[S_i]^{n_i}}{([S_i] + K_{mi})^{n_i}}$$

where $[S_i]$ is the amount of substrate $i$ available, $n_i$ is the [stoichiometric coefficient](https://en.wikipedia.org/wiki/Chemical_equation#Structure) of substrate $i$, $K_{mi}$ is the Michaelis constant for substrate $i$.
What exactly these values are is encoded in the domain itself (see [Genetics](#genetics)).

_Transporter domains_ essentially work in the same way. They are defined as $[A_{int}] \rightleftharpoons [A_{ext}]$ where $A_{int}$ is a molecule species $A$ inside the cell and $A_{ext}$ is the same molecule species outside the cell.

_Allosteric domains_ are also described by the same kinetic. However, they don't have $V_{max}$.
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

This simulation is implemented with millions of time steps in mind.
I believe that most interesting behaviors emerge after many time steps and that it doesn't take a high degree
of complexity for each single cell to create such behaviors.
Thus, this simulation is not trying to be a physically accurate depiction of processes in the cell.
It is a tradeoff between a reasonable amount of reality and performance.

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
And so deconstruction rates of this molecule species decrease.
Furthermore, as the ratio of products to substrates gets too high, the reaction stops or turns around
(free Gibbs energy of the reaction as described in [Energy](#energy)).
So, generally proteins shouldn't attempt to deconstruct more substrate than possible.
However, if a protein has a very high $V_{max}$ and a very low $K_M$ it can happen that
during one step it would deconstruct more substrate than actually available.
This can also happen if multiple proteins in the same cell would deconstruct the same molecule species.

To avoid deconstructing more substrate than available and creating negative molecule abundances there is a safety mechanism.
Each protein velocity $v$ is limited to the substrate molecule species that the protein deconstructs (accounting for the stoichiometric coefficient).
If a protein in any cell would by its normal kinetics deconstruct more substrate than available in that cell, it would be slowed down.
If there are multiple proteins deconstructing the same molecule in a cell, all of these proteins are slowed down
by the same factor (accounting for the stoichiometric coefficient).
Thus, a protein would never be able to create negative substrate abundances.
See `Kinetics.integrate_signals` in [magicsoup/kinetics.py](./magicsoup/kinetics.py) for more information.

In fact, proteins are slowed down a bit more so that a small amount of substrate $\epsilon > 0$ if left.
This small value $\epsilon$ is set in [magicsoup/constants.py](./magicsoup/constants.py).
Molecule abundances should not reach exact $0.0$ since this would create infinite values during $Q$ calculation.
$\epsilon$ exists to avoid that.
As a side effect, $\epsilon$ also serves as a hint for world molecule setups.
_Normal molecule_ abundances should be a few orders of magnitude greater than $\epsilon$.
Otherwise kinetics would be dominated by this cutoff.

#### High molecule abundances

On the other hand, it is possible that protein kinetics generate infinite numbers (not from zero abundances).
This can happen in proteins that catalyze reactions with high stoichiometric coefficients and a high
number of substrate or product species.
During `Kinetics.integrate_signals` in [magicsoup/kinetics.py](./magicsoup/kinetics.py)
stoichiometric coefficients become exponents and all product or substrate species are furthermore multiplied with each other.
If molecule abundances are very high this can overflow the default single precision floating point.
Therefore, molecule abundances and $V_{max}$ ranges should not be set too high.
From trial and error I can recommend to keep these values $\leq 10$.

If one wants to increase protein $V_{max}$ beyond that, it is better to rather do multiple
`integrate_signals` steps. _E.g._ calling `integrate_signals` 10 times in every time step
effectively multiplies the proteins $V_{max}$ by 10 without generating numerical instability.
This strategy would also give `integrate_signals` more continuous and realistic bahavior.
