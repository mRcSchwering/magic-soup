# Tutorials


## Simple CO2 Fixing Experiment

As a simple example one could try to teach cells to fix CO2 in a simulation.
Cells should create a set of proteins that can fix CO2 into some biologically useful form.
We could define acetyl-CoA as the desired end product.
So, a cell's survival will be based on its intracellular acetyl-CoA concentration and CO2 will be supplied in abundance.

### Chemistry

The basis for how cells are allowed to achieved that is defined by this simulation's chemistry.
Here, we will use the [Wood-Ljungdahl pathway](https://en.wikipedia.org/wiki/Wood%E2%80%93Ljungdahl_pathway) as a starting point.
There are a few molecule species and reactions that eventually end up in acetylating coenzyme A.
Below, we create a file _chemistry.py_ in which we define all these molecules and reactions.
For the sake of brevity some steps in the carbonyl branch were aggregated.

```python
# chemistry.py
from magicsoup.containers import Molecule, Chemistry

NADPH = Molecule("NADPH", 200.0 * 1e3)
NADP = Molecule("NADP", 100.0 * 1e3)
ATP = Molecule("ATP", 100.0 * 1e3)
ADP = Molecule("ADP", 70.0 * 1e3)

methylFH4 = Molecule("methyl-FH4", 360.0 * 1e3)
methylenFH4 = Molecule("methylen-FH4", 300.0 * 1e3)
formylFH4 = Molecule("formyl-FH4", 240.0 * 1e3)
FH4 = Molecule("FH4", 200.0 * 1e3)
formiat = Molecule("formiat", 20.0 * 1e3)
co2 = Molecule("CO2", 10.0 * 1e3, diffusivity=1.0, permeability=1.0)

NiACS = Molecule("Ni-ACS", 200.0 * 1e3)
methylNiACS = Molecule("methyl-Ni-ACS", 300.0 * 1e3)
HSCoA = Molecule("HS-CoA", 200.0 * 1e3)
acetylCoA = Molecule("acetyl-CoA", 260.0 * 1e3)


MOLECULES = [
    NADPH,
    NADP,
    ATP,
    ADP,
    methylFH4,
    methylenFH4,
    formylFH4,
    FH4,
    formiat,
    co2,
    NiACS,
    methylNiACS,
    HSCoA,
    acetylCoA,
]

REACTIONS = [
    ([NADPH], [NADP]),
    ([ATP], [ADP]),
    ([co2], [formiat]),
    ([formiat, FH4], [formylFH4]),
    ([formylFH4], [methylenFH4]),
    ([methylenFH4], [methylFH4]),
    ([methylFH4, NiACS], [FH4, methylNiACS]),
    ([methylNiACS, co2, HSCoA], [NiACS, acetylCoA]),
]
```

Each molecule species was created with a unique name and an energy of formation.
This energy has effects on reaction energies (more on this later # TODO).
Any number of molecules in this simulation is expressed in _mol_
and for this energy of formation it makes sense to think of it in terms of _J/mol_.
So, here _ATP_ is defined with _100 kJ/mol_.
Except for _CO2_ all defaults are kept.
For _CO2_ permeability and diffusivity is increased to account for the fact that
it diffuses rapidly and can permeate through cell membranes.

The reactions are tuples of lists of these molecule species.
The first tuple entry defined all substrates, the second all products.
A stoichiometric number >1 can be expressed by listing the molecule species multiple times.
Any reaction can happen in both directions, so tt is not necessary to define the reverse reaction.
In which direction a reaction will progress depends on its reaction energy and quotient.

### Setup

Eventually, we want to create [Chemistry][magicsoup.containers.Chemistry]
and [World][magicsoup.world.World] and then setp through time by repetitively calling different functions.
These functions would let the cells catalyze reactions and transport molecules,
diffuse and permeate molecules, kill cells, replicate cells, and create mutations.
Such a setup is shown below as _main.py_.
However, some functions are not implemented yet.

```python
# main.py
import magicsoup as ms
from .chemistry import REACTIONS, MOLECULES

def add_molecules(...):
    ...

def add_cells(...):
    ...

def kill_cells(...):
    ...

def replicate_cells(...):
    ...

def mutate_cells(...):
    ...

def main():
    chemistry = ms.Chemistry(reactions=REACTIONS, molecules=MOLECULES)
    world = ms.World(chemistry=chemistry)
    
    for _ in range(10_000):
        add_molecules()
        add_cells()
        world.enzymatic_activity()
        world.diffuse_molecules()
        world.degrade_molecules()
        kill_cells()
        replicate_cells()
        mutate_cells()
        world.increment_cell_survival()

if __name__ == "__main__":
    main()
```

Here, we would let the simulation run for 10k steps.
In each step a certain number of cells is added, then cells can work and molecules can diffuse,
and then cells are being killed or replicated based on their status.
Towards the end of the step all cells experience mutations and finnally -- for monitoring purposes --
the age of all surviving cells gets incremented by 1.

### Adding molecules

When [World][magicsoup.world.World] is instantiated by default it fills the map with molecules
of all molecule species to an average concentration of 10.
So, there are already some molecules.
However, we want to bring the cells to fix CO2, and if they are successful they will consume CO2.
So, we have to regularly supply CO2.

Additionally, the cells will need energy.
Using CO2 to acetylate CoA takes up energy.
For now, cells have some energy in the form of ATP and NADPH.
But once they figure out how to use it they wil quickly convert all ATP to ADP and all NADPH to NADP.
In our setup the cells have no means to restore these high energy molecules.
So, the world map has a certain energy stored that will run out as some point.
To give the cells ample time to develop, we can regularly replenish these energy carriers.

One naive approach would be to add CO2, ATP, and NADPH every round.
However, then over time these molecule concentrations would explode.
To avoid that we can be a bit more careful.
Here is another approach:

```python
def add_molecules(world: ms.World, co2: int, atp: int, adp: int, nadph: int, nadp: int):
    # keep NADPH/NADP and ATP/ADP ratios high
    for high, low in [(atp, adp), (nadph, nadp)]:
        high_avg = world.molecule_map[high].mean().item()
        low_avg = world.molecule_map[low].mean().item()
        if high_avg / (low_avg + 1e-4) < 5.0:
            world.molecule_map[high] += world.molecule_map[low] * 0.99
            world.molecule_map[low] *= 0.01

    # add CO2 up to a certain amount
    if world.molecule_map[co2].mean() < 50.0:
        world.molecule_map[co2] += 10.0
```

Energy carriers are converted and always kept in a high high-to-low energy ratio.
_I.e._ the same amount that is added to ATP, is substracted from ADP.
The ratios are always kept above 5, so that $ATP \rightleftharpoons ADP$
and $NADPH \rightleftharpoons NADP$ will always be a fast reactions.
CO2 on the other handside is constantly added.
To avoid infinitly growing CO2 concentrations we stop adding CO2
when concentrations are above 50.
When CO2 is on average 50 while other molecule species are on average 10,
reactions that use up CO2 will be favoured by the reaction quotient.

### Adding cells

So far, there are no cells yet.
Cells can be added through [add_random_cells()][magicsoup.world.World.add_random_cells]
by providing genomes.
They will be placed in random positions on the map and take up half the molecules
that were on that position.
There is a helper function `random_genome()` that can be used to generate genomes of a certain size.

From the setup above it is already apparent that cells will be added regularly.
One could add cells only once and then start the simulation, but it is very likely that they just all die
after a few time steps.
With completely random starting genomes, most cells will be inviable.
They will live for a few rounds and then die.
So, we want to keep adding cells until some viable ones appear.
To not gradually fill up the map with inviable cells we can just keep a certain amount of cells on the map.

```python
def add_cells(world: ms.World):
    dn = 1000 - len(world.cells)
    if dn > 0:
      genomes = [ms.random_genome(s=500) for _ in range(dn)]
      world.add_random_cells(genomes=genomes)
```

In the example above up to 1000 cells with random genomes are added every round.
If there are more than 1000 cells, no cell will be added anymore.
Usually, once a cell is viable and starts growing a colony, overall cell count will quickly exeed 1000.
It would also make sense to parametrize the maximum number of cells added and the genome size (here 500).
They are good for trimming the simulation.

### Replicating and killing cells

These are the main levers for exerting evolutionary pressure.
Most time will go into fine-tuning them.
Generally, we want to slowly increase or decrease the likelihood of cells dying or replicating over a certain variable (more on this later # TODO).
Here, these variables are intracellular concentration of specific molecule species.

Let's start with killing cells.
Which molecule species we look at for killing cells is really just trial-and-error.
It makes sense to choose something that would not contradict the replication criterion.
Here, we will kill cells with low NADPH and ATP concentrations.
Since NADPH and ATP concentrations are high anyway this should initially be easy for cells to avoid.
A helper function for sampling is defined.
Cells are killed with [kill_cells()][magicsoup.world.World.kill_cells] by providing their indexes.

```python
def sample(p: torch.Tensor) -> list[int]:
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()

def kill_cells(world: ms.World, atp: int, nadph: int):
    # low ATP
    x0 = world.cell_molecules[:, atp]
    idxs0 = sample(.5**4 / (.5**4 + x0**4))

    # low NADPH
    x1 = world.cell_molecules[:, nadph]
    idxs1 = sample(.5**4 / (.5**4 + x1**4))
    
    world.kill_cells(cell_idxs=list(set(idxs0 + idxs1)))
```

Cell replication should be based on acetyl-CoA.
Here, we could make the cell invest some energy in form of acetyl-CoA by converting it back to HS-CoA (taking away the acetyl group).
That way a cell has to continuously produce acetyl-CoA in order to replicate.
The better it can do that, the more frequent it will replicate.
Cells can be replicated with [replicate_cells()][magicsoup.world.World.replicate_cells] by supplying their indexes.
The indexes of successfully replicated cells are returned (if the cell has no space to replicate, it will not do so).

```python
def replicate_cells(world: ms.World, aca: int, hca: int):
    x = world.cell_molecules[:, aca]
    chosen = sample(x**5 / (x**5 + 15.0**5))
    
    allowed = torch.argwhere(world.cell_molecules[:, aca] > 2.0).flatten().tolist()
    idxs = list(set(chosen) & set(allowed))

    replicated = world.replicate_cells(parent_idxs=idxs)
    if len(replicated) > 0:
        parents, children = list(map(list, zip(*replicated)))
        world.cell_molecules[parents + children, aca] -= 1.0
        world.cell_molecules[parents + children, hca] += 1.0
```

Here, we use a function to sample replicating cells based on acetyl-CoA concentrations.
For all replicated cells 2 mol acetyl-CoA should then be converted to HS-CoA.
That means a cell that was chosen to replicate but doesn't have at least 2 mol acetyl-CoA cannot replicate.
After a cell sucessfully replicated its molecules are equally shared with new cell (called _child_ here).
To perform the conversion 1 mol of acetyl-CoA is removed and 1 mol of HS-CoA is added for both cells.

### Mutating cells

To continously create variation among cells they are all mutated at every step.
There are some functions for mutating genomes already provided in the `ms.mutations` module.
But anything that mutated strings can be used.
Here, I am using [point_mutations()][magicsoup.mutations.point_mutations] to apply random
point mutations with a rate of 1e-3 per nucleotide. 10% of them are InDels.

```python
def mutate_cells(world: ms.World):
    mutated = ms.point_mutations(seqs=[d.genome for d in world.cells])
    world.update_cells(genome_idx_pairs=mutated)
```

[update_cells()][magicsoup.world.World.update_cells] is used to update the cells whose genome was altered.
This function derives each cell's new proteome and does all required updates.
It is currently the performance bottleneck, so it's best to only provide the cells that were really altered
(don't always recalculate all proteomes).

### Putting it together

Finally, we can combine everything in _main.py_.
In the functions above I always used indexes to reference certain molecule species on
`world.molecule_map` and `world.cell_molecules`. Those indexes are derived from [Chemistry][magicsoup.containers.Chemistry].
Molecule species are always ordered in the same way as on `chemistry.molecules`.
Below, I am creating a name-to-index mapping for these molecule with `mol_2_idx = {d.name: i for i, d in enumerate(chemistry.molecules)}`
after [Chemistry][magicsoup.containers.Chemistry] is instantiated.

```python
# main.py
import torch
import magicsoup as ms
from .chemistry import REACTIONS, MOLECULES


def add_molecules(world: ms.World, co2: int, atp: int, adp: int, nadph: int, nadp: int):
    # keep NADPH/NADP and ATP/ADP ratios high
    for high, low in [(atp, adp), (nadph, nadp)]:
        high_avg = world.molecule_map[high].mean().item()
        low_avg = world.molecule_map[low].mean().item()
        if high_avg / (low_avg + 1e-4) < 5.0:
            world.molecule_map[high] += world.molecule_map[low] * 0.99
            world.molecule_map[low] *= 0.01

    # add CO2 up to a certain amount
    if world.molecule_map[co2].mean() < 50.0:
        world.molecule_map[co2] += 10.0


def add_cells(world: ms.World):
    dn = 1000 - len(world.cells)
    if dn > 0:
        genomes = [ms.random_genome(s=500) for _ in range(dn)]
        world.add_random_cells(genomes=genomes)


def sample(p: torch.Tensor) -> list[int]:
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()


def kill_cells(world: ms.World, atp: int, nadph: int):
    # low ATP
    x0 = world.cell_molecules[:, atp]
    idxs0 = sample(0.5**4 / (0.5**4 + x0**4))

    # low NADPH
    x1 = world.cell_molecules[:, nadph]
    idxs1 = sample(0.5**4 / (0.5**4 + x1**4))

    world.kill_cells(cell_idxs=list(set(idxs0 + idxs1)))


def replicate_cells(world: ms.World, aca: int, hca: int):
    x = world.cell_molecules[:, aca]
    chosen = sample(x**5 / (x**5 + 15.0**5))

    allowed = torch.argwhere(world.cell_molecules[:, aca] > 2.0).flatten().tolist()
    idxs = list(set(chosen) & set(allowed))

    replicated = world.replicate_cells(parent_idxs=idxs)
    if len(replicated) > 0:
        parents, children = list(map(list, zip(*replicated)))
        world.cell_molecules[parents + children, aca] -= 1.0
        world.cell_molecules[parents + children, hca] += 1.0


def mutate_cells(world: ms.World):
    mutated = ms.point_mutations(seqs=[d.genome for d in world.cells])
    world.update_cells(genome_idx_pairs=mutated)


def main():
    chemistry = ms.Chemistry(reactions=REACTIONS, molecules=MOLECULES)
    world = ms.World(chemistry=chemistry)

    mol_2_idx = {d.name: i for i, d in enumerate(chemistry.molecules)}
    CO2_IDX = mol_2_idx["CO2"]
    ATP_IDX = mol_2_idx["ATP"]
    ADP_IDX = mol_2_idx["ADP"]
    NADPH_IDX = mol_2_idx["NADPH"]
    NADP_IDX = mol_2_idx["NADP"]
    ACA_IDX = mol_2_idx["acetyl-CoA"]
    HCA_IDX = mol_2_idx["HS-CoA"]

    for _ in range(10_000):
        add_molecules(
            world=world,
            co2=CO2_IDX,
            atp=ATP_IDX,
            adp=ADP_IDX,
            nadph=NADPH_IDX,
            nadp=NADP_IDX,
        )
        add_cells(world=world)
        world.enzymatic_activity()
        world.diffuse_molecules()
        world.degrade_molecules()
        kill_cells(world=world, atp=ATP_IDX, nadph=NADPH_IDX)
        replicate_cells(world=world, aca=ACA_IDX, hca=HCA_IDX)
        mutate_cells(world=world)
        world.increment_cell_survival()

if __name__ == "__main__":
    main()
```

## Handling

These are some examples for monitoring, checkpointing, and parametrizing simulations.
Let's assume a setup like described in the [experiment above](#simple-co2-fixing-experiment).
So, the _main.py_ looks something like this:

```python
# main.py
import torch
import magicsoup as ms
from .chemistry import REACTIONS, MOLECULES

...

def main():
    chemistry = ms.Chemistry(reactions=REACTIONS, molecules=MOLECULES)
    world = ms.World(chemistry=chemistry)
    ...

    for _ in range(10_000):
        ...

if __name__ == "__main__":
    main()
```

### Monitoring

One nice tool for monitoring an ongoing simulation run is [TensorBoard](https://www.tensorflow.org/tensorboard).
It's an app that watches a directory of your run and displays logged data as line charts, histograms, images and more.
It can be installed from [pipy](https://pypi.org/project/tensorboard/).
_pytorch_ already includes a `SummaryWriter` that can be used for writing these logging files.

```python
# main.py
import datetime as dt
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import magicsoup as ms
from .chemistry import REACTIONS, MOLECULES

THIS_DIR = Path(__file__).parent
...

def main():
    now = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
    writer = SummaryWriter(log_dir=THIS_DIR / "runs" / now)

    chemistry = ms.Chemistry(reactions=REACTIONS, molecules=MOLECULES)
    world = ms.World(chemistry=chemistry)
    ...

    for _ in range(10_000):
        ...

if __name__ == "__main__":
    main()
```

When it is instantiated it creates `log_dir` if it doesn't already exist.
This is where all the logging files will go.
Add `runs/` to `.gitignore` to avoid committing this directory.
In the example above, I am also adding the current date and time as a a subdirectory,
so that you can start a run multiple times without overriding the previous ones.

How to use the `SummaryWriter` is explained in [the docs](https://pytorch.org/docs/stable/tensorboard.html).
It supports a few data types.
We will start with recording some scalars about cell growth.
Additionally, we can visualize the cell map by taking a picture of it.
These pictures are a bit heavy, so we will only capture one every 10 steps. 

```python
# main.py
import datetime as dt
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import magicsoup as ms
from .chemistry import REACTIONS, MOLECULES

THIS_DIR = Path(__file__).parent
...

def write_scalars(world: ms.World, writer: SummaryWriter, step: int):
    writer.add_scalar("Cells/total[n]", len(world.cells), step)
    writer.add_scalar("Cells/Survival[avg]", world.cell_survival.mean(), step)
    writer.add_scalar("Cells/Divisions[avg]", world.cell_divisions.mean(), step)

def write_images(world: ms.World, writer: SummaryWriter, step: int):
    writer.add_image("Maps/Cells", world.cell_map, step, dataformats="WH")

def main():
    now = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
    writer = SummaryWriter(log_dir=THIS_DIR / "runs" / now)

    chemistry = ms.Chemistry(reactions=REACTIONS, molecules=MOLECULES)
    world = ms.World(chemistry=chemistry)
    ...

    for step in range(10_000):
        ...
        write_scalars(world=world, writer=writer, step=step)
        if step % 10 == 0:
            write_images(world=world, writer=writer, step=step)

if __name__ == "__main__":
    main()
```

There is a pattern to labelling the variables on how they will be displayed in the app.
_E.g._ there will be a _Cells_ and a _Maps_ section.
The image dataformat is `WH` because dimension 0 of `world.cell_map` represents the x axis,
and dimension 1 the y axis.
You can start the app by pointing it at the runs directory `tensorboard --logdir=./runs`.

### Parameters

You might want to parametrize _main.py_ so that you can start it with different conditions.
Let's say we want to parametrize the number of steps: sometimes we just want to run a few steps to see if it works,
sometimes we want to start a long run with thousands of steps.
There are many tools for that.
I am going to stick to the standard library and use [argparse](https://docs.python.org/3/library/argparse.html).

```python
# main.py
import json
import datetime as dt
from pathlib import Path
from argparse import ArgumentParser, Namespace
import torch
import magicsoup as ms
from .chemistry import REACTIONS, MOLECULES

THIS_DIR = Path(__file__).parent
...

def main(args: Namespace):
    outdir = THIS_DIR / "runs" / dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
    outdir.mkdir(exist_ok=True, parents=True)
    with open(outdir / "hparams.json", "w") as fh:
        json.dump(vars(parsed_args), fh)

    chemistry = ms.Chemistry(reactions=REACTIONS, molecules=MOLECULES)
    world = ms.World(chemistry=chemistry)
    ...

    for _ in range(args.n_steps):
        ...

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_steps", default=10_000, type=int)
    parsed_args = parser.parse_args()
    main(args=parsed_args)
```

Now every run can be started with a diffent `n_steps`, so it makes sense to note
parameters somewhere when starting a new run.
Here, I am doing this by just writing `var(parsed_args)` to JSON.
This can also be done with [SummaryWriter.add_hparams](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_hparams).
As in the TensorBoard example above I am creating a _runs_ directory with the current date and time.

### Checkpoints

[World][magicsoup.world.World] has some functions for saving (and loading) itself.
[save()][magicsoup.world.World.save] is used to save the whole world object as pickle file.
It can be restored using [from_file()][magicsoup.world.World.from_file].
However, during the simulation not everything on the world object changes.
A smaller and quicker way to save is [save_state()][magicsoup.world.World.save_state].
It only saves the parts which change when running the simulation (will write a few `.pt` files).
A state can be restored with [load_state()][magicsoup.world.World.load_state].
So, in the beginning one [save()][magicsoup.world.World.save] is needed to save the whole object.
Then, [save_state()][magicsoup.world.World.save_state] can be used to save a certain time point.

```python
# main.py
import datetime as dt
from pathlib import Path
import torch
import magicsoup as ms
from .chemistry import REACTIONS, MOLECULES

THIS_DIR = Path(__file__).parent
...

def main():
    outdir = THIS_DIR / "runs" / dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
    outdir.mkdir(exist_ok=True, parents=True)

    chemistry = ms.Chemistry(reactions=REACTIONS, molecules=MOLECULES)
    world = ms.World(chemistry=chemistry)
    world.save(rundir=outdir)
    ...

    for step in range(10_000):
        ...
        if step % 100 == 0:
            world.save_state(statedir=outdir / f"step={step}")

if __name__ == "__main__":
    main()
```

As in the examples above I am creating a _runs_ directory with the current date and time.
I am also not saving every step to reduce the time spend saving and the size of _runs/_.