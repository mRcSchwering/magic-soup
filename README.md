## Magicsoup

This is a game that simulates cell metabolic and transduction pathway evolution.
Define a 2D world with certain molecules and reactions.
Add a few cells and create evolutionary pressure by selectively replicating and killing them.
Then run and see what random mutations can do.

![monitoring run](https://raw.githubusercontent.com/mRcSchwering/magic-soup/main/tensorboard_example.png)
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
