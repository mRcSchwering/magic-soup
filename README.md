## MagicSoup

---

**Documentation**: [https://magic-soup.readthedocs.io/](https://magic-soup.readthedocs.io/)

**Source Code**: [https://github.com/mRcSchwering/magic-soup](https://github.com/mRcSchwering/magic-soup)

**PyPI**: [https://pypi.org/project/magicsoup/](https://pypi.org/project/magicsoup/)

---

This game simulates cell metabolic and transduction pathway evolution.
Define a 2D world with molecules and reactions.
Add a few cells and create evolutionary pressure by selectively replicating and killing them.
Then run and see what random mutations can do.

![random cells](https://raw.githubusercontent.com/mRcSchwering/magic-soup/main/docs/img/animation[small].gif)

_Cell growth of 1000 cells with different genomes was simulated. Top row: Cell maps showing all cells (all) and the 3 fastest growing celllines (CL0-2). Middle and bottom rows: Development of total cell count and molecule concentrations over time._

## Installation

For CPU alone you can just do:

```bash
pip install magicsoup
```

This simulation relies on [PyTorch](https://pytorch.org/).
You can move almost all calculations to a GPU.
This increases performance a lot and is recommended.
In this case first setup PyTorch (>=2.0.0,<2.2.0) with CUDA as described in [Get Started (pytorch.org)](https://pytorch.org/get-started/locally/),
then install MagicSoup afterwards.


## Example

The basic building blocks of what a cell can do are defined by the world's [chemistry][magicsoup.containers.Chemistry].
There are [molecules][magicsoup.containers.Molecule] and reactions that can convert these molecules.
Cells can develop proteins with domains that can catalyze reactions, transport molecules and be regulated by them.
Reactions and transports always progress into the energetically favourable direction.
Below, I am defining a chemistry with reaction CO2 + NADPH $\rightleftharpoons$ formiat + NADP | -90 kJ.

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
Here, I am generating 100 cells with random genomes of 500 base pairs each and place them
randomly on a 2D world map.

```python
genomes = [ms.random_genome(s=500) for _ in range(100)]
world.spawn_cells(genomes=genomes)
```

Cells discover new proteins by chance through mutations.
In the function below all cells experience 0.0001 point mutations per base pair.
Then, neighbouring cells have a chance to exchange parts of their genome.

```python
def mutate_cells(world: ms.World):
    world.mutate_cells(p=1e-4)
    world.recombinate_cells(p=1e-6)
```

Evolutionary pressure can be applied by selectively killing or replicating cells.
Here, cells have an increased chance of dying when formiat concentrations decrease
and an increased chance of replicating when formiat concentrations increase.

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

Finally, the simulation itself is run in a loop by repetitively calling different methods.
With [enzymatic_activity()][magicsoup.world.World.enzymatic_activity] chemical reactions and molecule transport
in cells advance by one time step.
[diffuse_molecules()][magicsoup.world.World.diffuse_molecules] lets molecules on the world map diffuse and permeate through cell membranes
by one time step.

```python
for _ in range(1000):
    world.enzymatic_activity()
    kill_cells(world=world)
    replicate_cells(world=world)
    mutate_cells(world=world)
    world.diffuse_molecules()
```

See the [Docs](https://magic-soup.readthedocs.io/) for more examples and a description of all the mechanics of this simulation

