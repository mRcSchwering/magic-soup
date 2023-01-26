# Magicsoup

This is a game that simulates cell metabolic and transduction pathway evolution.
Define a 2D world with certain molecules and reactions.
Add a few cells and create evolutionary pressure by selectively replicating and killing them.
Then run and see what random mutations can do.

![monitoring run](https://raw.githubusercontent.com/mRcSchwering/magic-soup/main/tensorboard_example.png)
_Watching an ongoing simulation using TensorBoard. In [this simulation](https://github.com/mRcSchwering/luca/tree/main/experiments/e1_co2_fixing) cells were made to fix CO2 from an artificial CO2 source in the middle of the map._

Proteins in this simulation are made up of catalytic, transporter, and regulatory domains.
They are energetically coupled within the same protein and mostly follow Michaelis-Menten-Kinetics.
Chemical reactions and molecule transport only ever happens in the energetically favourable direction, as defined by the Nernst equation.
With enough proteins a cell is able to create complex networks with cascades, feed-back loops, and oscillators.
Through these networks a cell is able to communicate with its environment and form relationships with other cells.
How many proteins a cell has, what domains they have, and how these domains are parametrized is all defined by its genome.
Through random mutations cells search this vast space of possible proteomes.
By allowing only certain proteomes to replicate, this search can be guided towards a specific goal.

## Concepts

In general, you create a [Chemistry][magicsoup.containers.Chemistry] object with reactions and molecule species.
For molecules you can define things like energy, permeability, diffusivity.
See the [Molecule][magicsoup.containers.Molecule] class for more info.
Reactions are just tuples of substrate and product molecule species.

Then, you create a [World][magicsoup.world.World] object which defines things like a map and genetics.
It carries all data that describes the world at this time step with cells, molecule distributions and so on.
On this object there are also methods that are used to advance the world by one time step.

Usually, you would only adjust `Molecule`s and `World`.
However, in some cases you might want to change the way how genetics work;
_e.g._ change the way how certain domains are encoded, or change how coding regions are defined.
In that case you can override the default [Genetics][magicsoup.genetics.Genetics] object.

Apart from that, you create the simulation to your own likings.
From the `World` object you can observe molecule contents in cells
and use that this information to kill or replicate them (like in the example above).
You can also alter parts of the world, like creating concentration gradients
or regularly supplying the world with certain molecules/energy.
The documentation of [World][magicsoup.world.World] describes all attributes that could be of interest.

All major work is done by [PyTorch](https://pytorch.org/) and can be moved to a GPU.
`World` has an argument `device` to control that.
Please see [CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html) on how to use it.
And since this simulation already requires [PyTorch](https://pytorch.org/), it makes sense
to use [TensorBoard](https://pytorch.org/docs/stable/tensorboard.html) to interactively monitor your ongoing simulation.
