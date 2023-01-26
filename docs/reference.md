## Reference

::: magicsoup.world

### Containers

This module defines some classes which 
store a state and don't contain a lot of logic.
[Molecule][magicsoup.containers.Molecule] is the most important one.
This class defines a molecule species
and for every simulation at least one molecule species needs to be defined.
Molecule species are then listed and grouped in reactions in [Chemistry][magicsoup.containers.Chemistry],
which is used when initializing [World][magicsoup.world.World].
The other classes in this module don't need to be instantiated by a user.

::: magicsoup.containers
    options:
      heading_level: 4

::: magicsoup.genetics

::: magicsoup.kinetics

### Mutations

This is a supporting module with some
functions for efficiently creating mutations
in nucleotide sequences.

::: magicsoup.mutations
    options:
      heading_level: 4
