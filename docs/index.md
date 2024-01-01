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
In the function below all cells experience 0.001 random point mutations per base pair.
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

You create a [Chemistry][magicsoup.containers.Chemistry] object with molecules and reactions.
For molecules you can define things like energy, permeability, diffusivity.
Reactions are tuples of substrate and product molecule species.
Then, a [World][magicsoup.world.World] object defines a world map with information about cells and molecule concentrations.
By manipulating attributes and calling methods on [World][magicsoup.world.World] the simulation incrementally advances time.
The [tutorials](./tutorials.md) section explains this with a simple simulation.

By manipulating [World][magicsoup.world.World] attributes different experimental setups can be simulated.
These could be realistic like [batch culture](./tutorials.md#passaging-cells),
a [Chemostat](./tutorials.md#manipulating-concentrations),
[genetic engineering](./tutorials.md#generating-genomes), or completely fantastical.
At any point in time any cell in the simulation can be [examined in detail](./tutorials.md#interpreting-genomes)
inlcuding its heredity, genome, proteome, how its molecule contants and environment.
Over time (steps) evolutionary processes can be analyzed in detail.

[Simulation mechanics](./mechanics.md) are inspired by procaryotes living in a 2D world.
They try to be performant, yet meaningful, without posing any restrictions on how proteomes can evolve.
All major computations are done using [PyTorch](https://pytorch.org/)
and can be [moved to a GPU](./tutorials.md#gpu-and-tensors).
Furthermore, [there are tools](./tutorials.md#managing-simulation-runs) for
monitoring, checkpointing, managing simulations. 

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
