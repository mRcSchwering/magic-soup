# Tutorials

## Simple Experiment

_MagicSoup_ is trying to only provide the simulation engine.
Everything else should be up to the user, so that any experimental setup can be created.
However, this also means a lot of code has to be written by yourself.
As an example let's try to teach cells to convert CO2 into acetyl-CoA.
Cell survival will be based on intracellular acetyl-CoA concentrations and CO2 will be supplied in abundance.

### Chemistry

The most important thing of our simulated world is the [Chemistry][magicsoup.containers.Chemistry] object.
It defines which [Molecules][magicsoup.containers.Molecule] exist and how they move around,
which reactions are possible and how much energy they release.

Here, we will use the [Wood-Ljungdahl pathway](https://en.wikipedia.org/wiki/Wood%E2%80%93Ljungdahl_pathway) as inspiration.
There are a few molecule species and reactions that eventually acetylate coenzyme A.
Below, we create a file _chemistry.py_ in which we define all these molecules and reactions.
For the sake of brevity some steps were aggregated.

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

Each molecule species was created with a unique name and an energy.
This energy has effects on reaction energies (more on this in [Molecule energies](#molecule-energies)).
So, here _ATP_ is defined with _100 kJ/mol_. Its hydrolysis to _ADP_ releases _30 kJ/mol_.
Except for _CO2_ all defaults are kept.
For _CO2_ permeability and diffusivity is increased to account for the fact that
it diffuses rapidly and can permeate through cell membranes.

The reactions are tuples of lists of these molecule species.
The first tuple entry defines all substrates, the second all products.
A stoichiometric number >1 can be expressed by listing the molecule species multiple times.
All reactions are reversible, so it is not necessary to define the reverse reaction.
In which direction a reaction will progress depends on its reaction energy and quotient.

### Setup

Eventually, we want to create a [Chemistry][magicsoup.containers.Chemistry]
and a [World][magicsoup.world.World] object and then step through time by repetitively calling different functions.
This is what our _main.py_ will look like:

```python
# main.py
import magicsoup as ms
from .chemistry import REACTIONS, MOLECULES

def prepare_medium(...):
    ...

def add_cells(...):
    ...

def activity(...):
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
    prepare_medium()
    add_cells()

    for _ in range(10_000):
        activity()
        kill_cells()
        replicate_cells()
        mutate_cells()

if __name__ == "__main__":
    main()
```

Here, we prepare the medium, add some cells, and let the simulation run for 10k steps.
In each step cells can catalyze reactions and molecules can diffuse.
Then cells are selectively killed and replicated.
Finally, all surviving cells can experience mutations which change their genomes and proteomes.

### Adding molecules

When [World][magicsoup.world.World] is instantiated by default it fills the map with molecules
of all molecule species to an average concentration of 10 mM.
Here, we add extra CO2 and energy.
Our cells don't have any mechanism for restoring these energy carriers.
Sooner or later they will run out of energy.

```python
def prepare_medium(world: ms.World, i_co2: int, i_atp: int, i_adp: int, i_nadph: int, i_nadp: int):
    world.molecule_map[i_atp] = 100.0
    world.molecule_map[i_adp] = 0.0
    world.molecule_map[i_nadph] = 100.0
    world.molecule_map[i_nadp] = 0.0
    world.molecule_map[i_co2] = 100.0
```

### Adding cells

So far, there are no cells yet.
Cells can be spawned with [spawn_cells()][magicsoup.world.World.spawn_cells]
by providing genomes.
They will be placed in random positions on the map and take up half the molecules
that were on that position.
There is a helper function `random_genome()` that can be used to generate genomes of a certain size.

```python
def add_cells(world: ms.World, size=500, n_cells=1000):
    genomes = [ms.random_genome(s=size) for _ in range(n_cells)]
    world.spawn_cells(genomes=genomes)
```

### Cell Activity

This function essentially increments the world by one time step (1s).
[enzymatic_activity()][magicsoup.world.World.enzymatic_activity] lets cels catalyze reactions and transport molecules,
[degrade_molecules()][magicsoup.world.World.degrade_molecules] degrades molecules everywhere,
[diffuse_molecules()][magicsoup.world.World.diffuse_molecules] lets molecules diffuse and permeate,
[increment_cell_lifetimes()][magicsoup.world.World.increment_cell_lifetimes] increments cell lifetimes by 1.

```python
def activity(world: ms.World):
    world.enzymatic_activity()
    world.degrade_molecules()
    world.diffuse_molecules()
    world.increment_cell_lifetimes()
```

### Replicating and killing cells

These are the main levers for exerting evolutionary pressure.
Generally, we want to slowly increase or decrease the likelihood of cells dying or replicating over a certain variable (more on this in [Selection](#selection)).
Here, these variables will be intracellular molecule concentrations.

For killing cells we can look at intracellular ATP concentrations.
If they are low, chances of being killed are increased.
Cells are killed with [kill_cells()][magicsoup.world.World.kill_cells] by providing their indexes.
I am using a simple sigmoidal $f(x) = x^n/(x^n + c^n)$ to map likelihoods.

```python
def sample(p: torch.Tensor) -> list[int]:
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()

def kill_cells(world: ms.World, i_atp: int):
    x = world.cell_molecules[:, i_atp]
    idxs = sample(1.0**7 / (1.0**7 + x**7))
    world.kill_cells(cell_idxs=idxs)
```

Cell replication should be based on acetyl-CoA.
Here, we could make the cell invest some energy in form of acetyl-CoA by converting it back to HS-CoA (taking away the acetyl group).
That way a cell has to continuously produce acetyl-CoA in order to replicate.
Cells can divide with [divide_cells()][magicsoup.world.World.divide_cells] by providing their indexes.
The indexes of successfully replicated cells are returned.

```python
def replicate_cells(world: ms.World, i_aca: int, i_hca: int, cost=2.0):
    x = world.cell_molecules[:, i_aca]
    sampled_idxs = _sample(x**5 / (x**5 + 15.0**5))

    can_replicate = world.cell_molecules[:, i_aca] > cost
    allowed_idxs = torch.argwhere(can_replicate).flatten().tolist()
    
    idxs = list(set(sampled_idxs) & set(allowed_idxs))
    replicated = world.divide_cells(cell_idxs=idxs)
    
    if len(replicated) > 0:
        descendants = [dd for d in replicated for dd in d]
        world.cell_molecules[descendants, i_aca] -= cost / 2
        world.cell_molecules[descendants, i_hca] += cost / 2
```

Here, I decided that the cost of dividing is 2 mol acetyl-CoA.
Thus, I am also checking that only cells with at least 2 mol acetyl-CoA are allowed to divide.
After division their descendants have half of the ancestor's molecules.
To pay the price of dividing each descendant now has to hydrolyse 1 mol acetyl-CoA.
Note, I am not doing this before the replication because a cell might not successfully replicate.
If a cell has no space to replicate, it will not do so.

### Mutating cells

To continously create variation among cells they are all mutated at every step.
On [World][magicsoup.world.World] there are some functions to efficiently create mutations
and update the cells whose genomes have changed.
Below, I am using [mutate_cells()][magicsoup.world.World.mutate_cells],
which creates point mutations, with a rate of 1e-4 mutations per base pair.
I also decided to let recombinate with other cells if they have already lived for more than 10 steps.
[recombinate_cells()][magicsoup.world.World.recombinate_cells] works by creating random strand breaks.
Here, it creates 1e-6 strand brakes per base pair.

```python
def mutate_cells(world: ms.World, old=10):
    world.mutate_cells(p=1e-4)
    is_old = torch.argwhere(world.cell_lifetimes > old)
    world.recombinate_cells(cell_idxs=is_old.flatten().tolist(), p=1e-6)
```

[mutate_cells()][magicsoup.world.World.mutate_cells] and
[recombinate_cells()][magicsoup.world.World.recombinate_cells] are really just convenience functions.
You can also create mutations by yourself by just editing the strings in `world.cell_genomes`
and then calling [update_cells()][magicsoup.world.World.update_cells] for the cells that have changed.

### Putting it all together

Finally, we can combine everything in _main.py_.
In the functions above I always used indexes to reference certain molecule species on
`world.molecule_map` and `world.cell_molecules`.
Those indexes are derived from [Chemistry][magicsoup.containers.Chemistry].
Molecule species are always ordered in the same way as on `chemistry.molecules`.
For convenience there are 2 dictionaries `chemistry.molname_2_idx` and `chemistry.mol_2_idx`
for getting molecule indexes for either molecule names or molecule objects.

```python
# main.py
import torch
import magicsoup as ms
from .chemistry import REACTIONS, MOLECULES

def prepare_medium(world: ms.World, i_co2: int, i_atp: int, i_adp: int, i_nadph: int, i_nadp: int):
    world.molecule_map[i_atp] = 100.0
    world.molecule_map[i_adp] = 0.0
    world.molecule_map[i_nadph] = 100.0
    world.molecule_map[i_nadp] = 0.0
    world.molecule_map[i_co2] = 100.0

def add_cells(world: ms.World, size=500, n_cells=1000):
    genomes = [ms.random_genome(s=size) for _ in range(n_cells)]
    world.spawn_cells(genomes=genomes)

def activity(world: ms.World):
    world.enzymatic_activity()
    world.degrade_molecules()
    world.diffuse_molecules()
    world.increment_cell_lifetimes()

def sample(p: torch.Tensor) -> list[int]:
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()

def kill_cells(world: ms.World, i_atp: int):
    x = world.cell_molecules[:, i_atp]
    idxs = sample(1.0**7 / (1.0**7 + x**7))
    world.kill_cells(cell_idxs=idxs)

def replicate_cells(world: ms.World, i_aca: int, i_hca: int, cost=2.0):
    x = world.cell_molecules[:, i_aca]
    sampled_idxs = _sample(x**5 / (x**5 + 15.0**5))

    can_replicate = world.cell_molecules[:, i_aca] > cost
    allowed_idxs = torch.argwhere(can_replicate).flatten().tolist()
    
    idxs = list(set(sampled_idxs) & set(allowed_idxs))
    replicated = world.divide_cells(cell_idxs=idxs)
    
    if len(replicated) > 0:
        descendants = [dd for d in replicated for dd in d]
        world.cell_molecules[descendants, i_aca] -= cost / 2
        world.cell_molecules[descendants, i_hca] += cost / 2

def mutate_cells(world: ms.World, old=10):
    world.mutate_cells(p=1e-4)
    is_old = torch.argwhere(world.cell_lifetimes > old)
    world.recombinate_cells(cell_idxs=is_old.flatten().tolist(), p=1e-6)

def main():
    chemistry = ms.Chemistry(reactions=REACTIONS, molecules=MOLECULES)
    world = ms.World(chemistry=chemistry)

    i_co2 = chemistry.molname_2_idx["CO2"]
    i_atp = chemistry.molname_2_idx["ATP"]
    i_adp = chemistry.molname_2_idx["ADP"]
    i_nadph = chemistry.molname_2_idx["NADPH"]
    i_nadp = chemistry.molname_2_idx["NADP"]
    i_aca = chemistry.molname_2_idx["acetyl-CoA"]
    i_hca = chemistry.molname_2_idx["HS-CoA"]

    prepare_medium(
        world=world,
        i_co2=i_co2,
        i_atp=i_atp,
        i_adp=i_adp,
        i_nadph=i_nadph,
        i_nadp=i_nadp,
    )
    add_cells(world=world)

    for _ in range(10_000):
        activity(world=world)
        kill_cells(world=world, atp=i_atp)
        replicate_cells(world=world, aca=i_aca, hca=i_hca)
        mutate_cells(world=world)

if __name__ == "__main__":
    main()
```

## GPU and Tensors

[PyTorch](https://pytorch.org/) is used a lot in this simulation.
When initializing [World][magicsoup.world.World] parameter `device` can be used to move most calculations to a GPU.
_E.g._ with `device="cuda"` the default CUDA device is used (see [pytorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)).
Using a GPU usually speeds up the simulation by more than 100 times.

To achieve this performance cells and molecules are not represented as python objects.
Instead [World][magicsoup.world.World] maintains python lists and [PyTorch Tensors](https://pytorch.org/docs/stable/tensors.html).
Some of those tensors where used in the [example above](#simple-experiment): `world.molecule_map` and `world.cell_molecules`.
All lists and tensors that can be used to interact with the simulation are listed in
the [World][magicsoup.world.World] class documentation.
To keep performance these tensors should not be moved back and forth between GPU and CPU during the simulation.

```python
# float tensors on GPU get modified (fast)
world.molecule_map[0] = 10.0
world.cell_molecules[0] += 1.0
world.cell_molecules[world.cell_divisions > 10, 3] -= 1.0

mask = world.cell_lifetimes > 10  # bool tensor on GPU (fast)
idx_tensor = torch.argwhere(mask)  # long tensor on GPU (fast)
idx_lst = idx_tensor.flatten().tolist()  # tolist() sends integers to CPU (slow)

values = world.cell_divisions.float()  # convert integer to float tensor on GPU (fast)
mean = values.mean()  # calculate mean as 1-item float tensor on GPU (fast)
value = mean.item()  # item() sends value to CPU (slow)
```

There is a [PyTorch tutorial](https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html)
for working with tensors but in general you should try to use [tensor methods](https://pytorch.org/docs/stable/torch.html).
If you are familiar with [NumPy](https://numpy.org/) this should come easy.
Equivalents for most ndarray methods also exist on torch tensors.

## Boilerplate

These are some examples for monitoring, checkpointing, and parametrizing simulations.
Let's assume a setup like described in the [experiment above](#simple-experiment).
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

One nice tool for monitoring an ongoing simulation is [TensorBoard](https://www.tensorflow.org/tensorboard).
It's an app that watches a directory and displays logged data as line charts, histograms, images and more.
It can be installed from [pipy](https://pypi.org/project/tensorboard/).
_PyTorch_ already includes a `SummaryWriter` that can be used for writing these logging files.

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
    writer.add_scalar("Cells/Total[n]", world.n_cells, step)
    writer.add_scalar("Cells/Survival[avg]", world.cell_lifetimes.mean(), step)
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

![](./img/tensorboard_example.png)
_Watching 2 scalars and 1 image while a simulation is running_

### Parameters

You might want to parametrize _main.py_ so that you can start it with different conditions.
Let's say we want to parametrize the number of steps: sometimes we just want to run a few steps to see if it works,
sometimes we want to start a long run with thousands of steps.
There are many tools for that.
I am going to stick to the standard library and use [argparse](https://docs.python.org/3/library/argparse.html).

```python
# main.py
import datetime as dt
from argparse import ArgumentParser
import torch
import magicsoup as ms
from .chemistry import REACTIONS, MOLECULES

...

def main(kwargs: dict):
    chemistry = ms.Chemistry(reactions=REACTIONS, molecules=MOLECULES)
    world = ms.World(chemistry=chemistry)
    ...

    for _ in range(kwargs["n_steps"]):
        ...

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n-steps", default=10_000, type=int)
    parsed_args = parser.parse_args()
    main(vars(parsed_args))
```

### Checkpoints

[World][magicsoup.world.World] has some functions for saving (and loading) itself.
[save()][magicsoup.world.World.save] is used to save the whole world object as pickle file.
It can be restored using [from_file()][magicsoup.world.World.from_file].
However, during the simulation not everything on the world object changes.
A smaller and quicker way to save is [save_state()][magicsoup.world.World.save_state].
It only saves the parts which change when running the simulation (will write a few `.pt` and `.fasta` files).
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

## Selection

As long as cells can replicate, they can evolve by natural selection.
Better adapted cells will be able to replicate faster or die slower.
In the [experiment above](#simple-experiment) we used some functions to
derive a probability for killing or replicating cells based on intracellular ATP and acetyl-CoA concentrations.

```python
def sample(p: torch.Tensor) -> list[int]:
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()

def kill_cells(world: ms.World, i_atp: int):
    x = world.cell_molecules[:, i_atp]
    idxs = sample(1.0**7 / (1.0**7 + x**7))
    world.kill_cells(cell_idxs=idxs)

def replicate_cells(world: ms.World, i_aca: int, i_hca: int, cost=2.0):
    x = world.cell_molecules[:, i_aca]
    sampled_idxs = _sample(x**5 / (x**5 + 15.0**5))

    can_replicate = world.cell_molecules[:, i_aca] > cost
    allowed_idxs = torch.argwhere(can_replicate).flatten().tolist()
    
    idxs = list(set(sampled_idxs) & set(allowed_idxs))
    replicated = world.divide_cells(cell_idxs=idxs)
    
    if len(replicated) > 0:
        descendants = [dd for d in replicated for dd in d]
        world.cell_molecules[descendants, i_aca] -= cost / 2
        world.cell_molecules[descendants, i_hca] += cost / 2
```

Here, I used $f_k(x) = 1^7 / (1^7 + x^7)$ and $f_r(x) = x^5 / (x^5 + 15^5)$.
How to come up with useful functions and parameters?

### Estimating useful rates

It helps to guess some useful parameters to start with.
Say we use functions of the form $f(x) = x^n/(x^n + c^n)$ to map molecule concentrations to likelihoods.
We could try to set $c$ and $n$ in a way that that we have a good dynamic range for 0 to 5mM of $x$. 
Below I modelled the chance of a cell being killed or replicated for particular sets of $n$ and $c$
in cells with constant X concentrations.

![](img/kill_repl_prob.png)
_Probability of cells with constant X concentrations dying or dividing at least once when the chance to die depends on molecule concentration X with $p(X) =(X^7 + 1)^{-1}$ and the chance to replicate depends on it with $p(X) = X^5 / (X^5 + 15^5)$._

These events are not independent.
If a cell replicates, there are more cells that can replicate.
If it dies, it cannot replicate anymore.
We can simulate how cells with different X concentrations would grow given these kill and replication probabilities:

![](img/sim_cell_growth.png)
_Simulated growth of cells with constant X concentrations when the chance to die depends on molecule concentration X with $p(X) =(X^7 + 1)^{-1}$ and the chance to replicate depends on it with $p(X) = X^5 / (X^5 + 15^5)$._

Eventually, you still have to try out a bit by just running simulation.
While trying to come up with a good set of parameters you might see one of these patterns:

- **Cells die before forming a colony** 
  If they immediately die, the kill rate is probably too high.
  If it takes them many steps to die (with some cells lingering around for a while),
  it is probably too hard to replicate.
  In that case increase the replication rate.  
- **Cells quickly overgrow the map**
  The kill rate is probably too low.
  Only cells at the edge of the growing colony had a chance to adapt.
  Once the map is fully overgrown adaption ceases.
  There might be a well adapted cell somewhere but it cannot replicate.
- **Cells create wavefront, then die** If cells can replicate quickly,
  but also die quickly, they generate a wavefront of dividing cells which walks
  over the map in a few waves and then perishes.
  They often don't have enough time to adapt before going extinct.

![](./img/supporting/cell_growth.gif)

_Example cell growth in 4 simulations over 1000 steps with different kill and replication rates. (Left) with moderately high kill rate and low replicaiton rate.(Middle-left) with high replication rate and low kill rate. (Middle-right) with high replication and kill rate. (Right) with moderate kill and replication rate. Cell map is black, cells are white, every 5th step was captured._

Ideally, cells struggle a bit to survive but not so much as to go extinct.
They should have some time to adapt and space to grow.
To keep cells in exponential growth phase indefinitely you can passage them.

### Passaging cells

In this simulation passaging cells would equate to selecting a few cells, killing the others,
creating fresh medium, then spreading the surviving cells.
This way the cells have new molecules and open space to grow.

```python
def split_cells(world: ms.World, split_ratio=0.3):
    keep_n = int(world.n_cells * split_ratio)
    kill_n = max(world.n_cells - keep_n, 0)
    idxs = random.sample(list(range(world.n_cells)), k=kill_n)
    world.kill_cells(cell_idxs=idxs)
    prepare_medium(world=world)
    world.reposition_cells(cell_idxs=list(range(world.n_cells)))
```

Passaging itself selects for cells which are most abundant during the time of the passage.
Let's take the example from above with kill and replication functions $p_k(X) =(X^7 + 1)^{-1}$ and $p_r(X) = X^5 / (X^5 + 15^5)$.
The plot below shows simulated cell growth, where cells were passaged with different ratios whenever the total number of cells exeeded 7k.
Gray areas represent the total number of cells, stacked bar charts show the cell type composition before the passage.
We have cell types with X concentrations of 3, 4, 5, and 6.
As you can see all cell types except the fastest growing cell type (with $X=6$) quickly disappear.

![](img/splitting_cells.png)
_Simulated growth of cells with different molecule concentrations X when the chance to die depends on molecule concentration X with $p(X) =(X^7 + 1)^{-1}$ and the chance to replicate depends on it with $p(X) = X^5 / (X^5 + 15^5)$. Cells are split at different split ratios whenever they exceed a total count of 7000. Gray area represents total cell count, bars represent cell type composition before the split._

(More examples in [figures 10](./figures.md#10-passaging))

## Genomes

### Genome size

In the [experiment above](#simple-experiment) 1000 cells were initially added.
They each had a random genome of 500 base pairs length.

```python
def add_cells(world: ms.World):
    genomes = [ms.random_genome(s=500) for _ in range(1000)]
    world.spawn_cells(genomes=genomes)
```

During the simulation these genomes can become shorter or longer due to random mutations.
If the selection process works (see [Selection](#selection)) cells tend to accumulate new genes.
On average the cell profits from gaining a new gene more than it does from losing a gene.
It makes sense to introduce another selection function that will penalize cells with too large genomes.
This could be done in `kill_cells`:

```python
def kill_cells(world: ms.World, atp: int):
    x0 = world.cell_molecules[:, atp]
    sizes = torch.tensor([len(d) for d in exp.world.cell_genomes])
    idxs0 = sample(1.0**7 / (1.0**7 + x0**7))
    idxs1 = sample(sizes**7 / (sizes**7 + 3_000.0**7))
    world.kill_cells(cell_idxs=list(set(idxs0 + idxs1)))
```

Without it cell genomes will become exeecingly large over the course of thousands of steps.
This will eventually slow down the simulation.
The plot below shows how the genome size affects the cells proteome.

![](img/genome_sizes.png)
_Distributions for proteins per genome, domains per protein, and coding nucleotides per nucleotide for different genome sizes with domain probability 0.01_

When [World][magicsoup.world.World] is initialized, it creates a [Genetics][magicsoup.genetics.Genetics] instance.
This instances carries the logic of mapping nucleotide sequences to proteomes
(see [Mechanics](./mechanics.md) and [Genetics][magicsoup.genetics.Genetics] for details).
Changing the frequency by which nucleotide sequences can encode domains, changes the composition of genomes.
By default all 3 domain types are encoded by 2 codons (6 nucleotides) and 1% of all 2-codon combinations encode for 1 of these domain types.

These parameters can be changed.
In [figures 1](./figures.md#1-genomes) there are some examples on how genome compositions
change when these parameters are changed.
If you want to change [Genetics][magicsoup.genetics.Genetics] for your simulation, you have to create your
own instance and assign it to [World][magicsoup.world.World]:

```python
world = ms.World(...)
world.genetics = ms.Genetics(...)
```

## Molecule energies

In the [experiment at the beginning](#simple-experiment) some molecule species were defined in _chemistry.py_ with energies.

```python
formiat = Molecule("formiat", 20.0 * 1e3)
co2 = Molecule("CO2", 10.0 * 1e3, diffusivity=1.0, permeability=1.0)
```

In this simulation a chemical reaction is regarded as the decomposition of substrates and the creation of products.
In which direction the reaction will progress is defined by its energy (and the world's temperature) and the reaction quotient.
There exists an equilibrium at which the reaction effectively stops (this is explained in detail in [Mechanics](./mechanics.md)).
In the example above $CO2 \rightleftharpoons formiat$ has a reaction energy of 10 kJ/mol and would create a equilibrium constant of roughly 0.02.
If the reaction energy would have been defined as only 10 J/mol, the equilibrium constant would be almost 1.0.

![](img/equilibrium_constants.png)
_Chemistries with energies of around 10 kJ/mol, 100 kJ/mol, and 200 kJ/mol were created and random proteins were generated. Log10 equilibrium constant distributions of reactions catalyzed by these proteins are shown_

With lower reaction energies reactions are more dirven by reaction quotients.
For energetically coupled transporter and catalytic domains this means transporters can power more reactions,
_i.e._ cells can make more use of concentration gradients.

(More examples in [figures 5](./figures.md#5-equilibrium-constants))

## Molecule maps

When instantiating [World][magicsoup.world.World] by default all molecule species will added to `world.molecule_map`, normally distributed,
with an average concentration of 10.
In the [experiment above](#simple-experiment) this made sense for most molecule species.
We adjusted concentrations for energy carriers ATP/ADP and NADPH/NADP, and for the carbon source CO2.
For adding CO2, one could also think of creating gradient like shown below.
This could for example introduce a spatial dependence for cell growth or implement a [ChemoStat](https://en.wikipedia.org/wiki/Chemostat).

![](img/gradients.png)
_Molecule species X is added to molecule map and allowed to diffuse  to create a 1D gradient (top) or 2D gradients (bottom)._

In the plot above, these gradients were created by adding CO2 to specific pixels on the map, and removing them from other other.
The gradients emerge through diffusion.
However, the gradient takes a few hundred steps to reach its equilibrium.
Care must be taken to use the correct device.

```python
n = int(world.map_size / 2)
device = world.molecule_map.device

ones = torch.ones((world.map_size, world.map_size))
linspace = torch.cat([
    torch.linspace(1.0, 100.0, n),
    torch.linspace(100.0, 1.0, n)
])
gradient = torch.einsum("xy,x->xy", ones, linspace)

world.molecule_map[co2] = gradient.to(device)
```

The code above creates the 1D gradient that was shown in the plot for CO2.
By effectively doing `gradient.to(world.molecule_map.device)` we make sure that
the created tensor will be send to the same device that `world.molecule_map` was on.

(More examples in [figures 4](./figures.md#4-molecule-diffusion-and-degradation))
