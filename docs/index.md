# Magicsoup

---

**Documentation**: [https://magic-soup.readthedocs.io/](https://magic-soup.readthedocs.io/)

**Source Code**: [https://github.com/mRcSchwering/magic-soup](https://github.com/mRcSchwering/magic-soup)

**PyPI**: [https://pypi.org/project/magicsoup/](https://pypi.org/project/magicsoup/)

---

This game simulates cell metabolic and transduction pathway evolution.
Define a 2D world with certain molecules and reactions.
Add a few cells and create evolutionary pressure by selectively replicating and killing them.
Then run and see what random mutations can do.

![random cells](./img/animation.gif)

_Cell growth of 1000 cells with different genomes was simulated. Top row: Cell maps showing all cells (all) and the 3 fastest growing celllines (CL0-2). Middle and bottom rows: Development of total cell count and molecule concentrations over time._

Proteins in this simulation are made up of catalytic, transporter, and regulatory domains.
They are energetically coupled within the same protein and mostly follow Michaelis-Menten-Kinetics.
Chemical reactions and molecule transport only ever happens in the energetically favourable direction.
With enough proteins a cell is able to create complex networks with cascades, feed-back loops, and oscillators.
Through these networks a cell is able to communicate with its environment and form relationships with other cells.
How many proteins a cell has, what domains they have, and how these domains are parametrized is all defined by its genome.
Through random mutations cells search this vast space of possible proteomes.
By allowing only certain genomes to replicate, this search can be guided towards a specific goal.

## Example

The basic building blocks of what a cell can do are defined by the world's [chemistry][magicsoup.containers.Chemistry].
There are [molecules][magicsoup.containers.Molecule] and reactions that can convert these molecules.
Cells can develop proteins with domains that can catalyze reactions, transport molecules and be regulated by them.
Reactions and transports always progress into the energetically favourable direction.
Below, I am defining a chemistry with reaction $\text{CO2} + \text{NADPH} \rightleftharpoons \text{formiat} + \text{NADP} | -90 \text{kJ}$.

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
randomly on the 2D world map.

```python
genomes = [ms.random_genome(s=500) for _ in range(100)]
world.spawn_cells(genomes=genomes)
```

Cells discover new proteins by chance through mutations.
In the function below all cells experience 0.001 random point mutations per nucleotide.
40% of them will be indels.

```python
def mutate_cells(world: ms.World):
    mutated = ms.point_mutations(seqs=world.cell_genomes)
    world.update_cells(genome_idx_pairs=mutated)
```

Evolutionary pressure can be applied by selectively killing or replicating cells.
Here, cells have an increased chance of dying when formiat gets too low
and an increased chance of replicating when formiat gets high.

```python
def sample(p: torch.Tensor) -> list[int]:
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()

def kill_cells(world: ms.World):
    x = world.cell_molecules[:, 2]
    idxs = sample(.01 / (.01 + x))
    world.kill_cells(cell_idxs=idxs)

def replicate_cells(world: ms.World):
    x = world.cell_molecules[:, 2]
    idxs = sample(x ** 3 / (x ** 3 + 20.0 ** 3))
    world.divide_cells(cell_idxs=idxs)
```

Finally, the simulation itself is run in a python loop by repetitively calling the different steps.
With [enzymatic_activity()][magicsoup.world.World.enzymatic_activity] chemical reactions and molecule transport
in cells advance by one time step.
[diffuse_molecules()][magicsoup.world.World.diffuse_molecules] lets molecules on the world map diffuse and permeate through cell membranes
(if they can) by one time step.

```python
for _ in range(1000):
    world.enzymatic_activity()
    kill_cells(world=world)
    replicate_cells(world=world)
    mutate_cells(world=world)
    world.diffuse_molecules()
    world.increment_cell_lifetimes()
```

## Concepts

In general, you create a [Chemistry][magicsoup.containers.Chemistry] object with molecules and reactions.
For molecules you can define things like energy, permeability, diffusivity.
See the [Molecule][magicsoup.containers.Molecule] class for more info.
Reactions are tuples of substrate and product molecule species.

Then, you create a [World][magicsoup.world.World] object which defines a world map.
It carries all information about cells and molecule concentrations at a particular point in time.
On this object there are some methods for advancing the simulation by one time step.
By default molecule concentrations are in mM, energies are in J/mol, and a time step represents 1s.

For customizing a simulation you can interact with the working attributes of the [World][magicsoup.world.World] object.
Like in the example above you could inspect the molecule concentrations in cells and
base your decision to replicate or kill a cell on this information.
You could also _e.g._ add molecules to the world by editing `world.molecule_map`,
or edit cell genomes using [update_cells()][magicsoup.world.World.update_cells].
The documentation of [World][magicsoup.world.World] describes all attributes that could be of interest.

All major computations are done using [PyTorch](https://pytorch.org/) and can be moved to a GPU.
[World][magicsoup.world.World] has an argument `device` to control that.
Please see [CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html) on how to use it.
And since this simulation already requires [PyTorch](https://pytorch.org/), it makes sense
to use [TensorBoard](https://pytorch.org/docs/stable/tensorboard.html) to interactively monitor your ongoing simulation.

## Installation

For CPU alone you can just do:

```bash
pip install magicsoup
```

This simulation relies on [PyTorch](https://pytorch.org/).
You can move almost all calculations to a GPU.
This increases performance a lot and is recommended.
In this case first setup PyTorch with CUDA as described in [Get Started (pytorch.org)](https://pytorch.org/get-started/locally/),
then install MagicSoup afterwards.
