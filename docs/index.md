# Magicsoup

---

**Documentation**: [https://magic-soup.readthedocs.io/](https://magic-soup.readthedocs.io/)

**Source Code**: [https://github.com/mRcSchwering/magic-soup](https://github.com/mRcSchwering/magic-soup)

**PyPI**: [https://pypi.org/project/magicsoup/](https://pypi.org/project/magicsoup/)

---

This is a simulation for cell metabolic and transduction pathway evolution.
Define a world with molecules and reactions.
Add a few cells and create evolutionary pressure by selectively replicating and killing them.
Then run and see what random mutations can do.

![random cells](./img/animation.gif)

_Cell growth of 1000 cells with different genomes was simulated. Top row: Cell maps showing all cells (all) and the 3 fastest growing celllines (CL0-2). Middle and bottom rows: Development of total cell count and molecule concentrations over time._

Proteins in this simulation are made up of catalytic, transporter, and regulatory domains.
They are energetically coupled within the same protein and mostly follow reversible Michaelis-Menten-Kinetics.
Chemical reactions and molecule transport always progress in the energetically favourable direction.
With enough proteins a cell is able to create complex networks including cascades, hubs and feed-back loops.
Through these networks a cell is able to communicate with its environment and form relationships with other cells.
How many proteins a cell has, what domains they have, and how these domains are parametrized is all defined by its genome.
Through random mutations cells search a vast space of possible proteomes.
By allowing only certain genomes to replicate, this search can be guided towards a specific goal.
See [Mechanics](./mechanics.md) for more details about how the simulation works.

## Example

The basic building blocks of what a cell can do are defined by the world's [chemistry][magicsoup.chemistry.Chemistry].
There are [molecules][magicsoup.chemistry.Molecule] and [reactions][magicsoup.chemistry.Reaction] that can convert these molecules.
Cells can develop proteins with domains that can catalyze reactions, transport molecules and be regulated by them.
Reactions and transports always progress into the energetically favourable direction.
Below, defines a chemistry with reaction CO2 + NADPH $\rightleftharpoons$ formiat + NADP.

```python
import torch
import magicsoup as ms

NADPH = ms.Molecule("NADPH", 62, half_life=3, diffusivity=4e-6, permeability=0)
NADP = ms.Molecule("NADP", 2, half_life=60, diffusivity=4e-6, permeability=0)
co2 = ms.Molecule("CO2", 10, half_life=-1, diffusivity=2e-5, permeability=1)
formiat = ms.Molecule("formiat", 77, half_life=-1, diffusivity=1.5e-5, permeability=1e-6)

co2_nadph_to_formiat_NADP = ms.Reaction([co2, NADPH], [formiat, NADP])

chemistry = ms.Chemistry(reactions=[co2_nadph_to_formiat_NADP], molecules=[NADPH, NADP, formiat, co2])
world = ms.World(chemistry=chemistry)
```

By coupling multiple domains within the same protein, energetically unfavourable actions
can be powered with the energy of energetically favourable ones.
These domains, their specifications, and how they are coupled in proteins, is all encoded in the cell's genome.
The following generates 100 cells with random genomes and place them randomly on a world map with some initial molecules.

```python
world.culture.molecule_map[:] = 10
genomes = [ms.random_genome(s=500) for _ in range(100)]
world.spawn_cells(genomes=genomes)
```

The simulation [advances in time][magicsoup.world.World.advance_time] by letting cells work and molecules diffuse.
Cells discover new proteins by chance through [mutations][magicsoup.world.World.mutate_cells].
Evolutionary pressure can be applied by selectively [killing][magicsoup.world.World.kill_cells] or [dividing][magicsoup.world.World.divide_cells] cells.
E.g.:

```python
for _ in range(1000):
    world.advance_time()
    world.kill_cells(mask=world.cells.molecules[:, 3] < 0.1)
    divided = world.divide_cells(mask=world.cells.molecules[:, 1] > 0.5)
    world.mutate_cells(mask=divided)
```

See [Tutorials](./tutorials.md) for more detailed tutorials.

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
