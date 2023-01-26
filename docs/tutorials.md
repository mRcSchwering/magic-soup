# Tutorials


## Simple CO2 Fixing Experiment

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
## Asd

asd

## Asdf

asd