# Reference

## magicsoup.world

This module defines the [World][magicsoup.world.World] class.
A [World][magicsoup.world.World] is the main API for controlling a simulation.
It stores a simulation's state and provides methods for advancing a simulation.
[World][magicsoup.world.World] itself is a faccade that uses other classes such as
[Kinetics][magicsoup.kinetics.Kinetics] and [Genetics][magicsoup.genetics.Genetics].

::: magicsoup.world

## magicsoup.containers

This module defines some data classes.
[Chemistry][magicsoup.containers.Chemistry] is the most important one.
A [Chemistry][magicsoup.containers.Chemistry] object is needed when initializing [World][magicsoup.world.World].
It describes [Molecules][magicsoup.containers.Molecule] and reactions of a simulation.
[Molecule][magicsoup.containers.Molecule] defines a molecule species and each reaction is a grouping of molecule species.

::: magicsoup.containers

## magicsoup.factories

This module defines classes for generating other objects.
Most importantly it defines the [GenomeFact][magicsoup.factories.GenomeFact] class
which can be used to generate genomes that encode desired proteomes.
Desired proteomes can be defines by describing domains using factory classes
[CatalyticDomainFact][magicsoup.factories.CatalyticDomainFact],
[TransporterDomainFact][magicsoup.factories.TransporterDomainFact],
[RegulatoryDomainFact][magicsoup.factories.RegulatoryDomainFact].

::: magicsoup.factories

## magicsoup.genetics

This module provides functions mainly for translation and transcription.
Most of it is defined on the [Genetics][magicsoup.genetics.Genetics] class.
This class is used by [World][magicsoup.world.World].
Usually, a user does't need to use this class directly.

::: magicsoup.genetics

## magicsoup.kinetics

This module provides functions mainly for reaction kinetics.
Most of it is defined on the [Kinetics][magicsoup.kinetics.Kinetics] class.
This class is used by [World][magicsoup.world.World].
Usually, a user does't need to use this class directly.

::: magicsoup.kinetics

## magicsoup.mutations

This is a supporting module with some
functions for efficiently creating mutations
in nucleotide sequence strings.

::: magicsoup.mutations


## magicsoup.util

Some more helper functions

::: magicsoup.util