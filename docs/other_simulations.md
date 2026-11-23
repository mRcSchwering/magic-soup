# Other Simulations

### Evo²Sim (Evolution of Evolution Simulator)

[GitHub repository](https://github.com/charlesrocabert/Evo2Sim) · [Model
description](https://charlesrocabert.github.io/software/)

Evo²Sim is probably the closest existing simulation to MagicSoup. Like
MagicSoup, it models individual cells on a 2D toroidal grid, with
circular genomes encoding enzymes and transcription factors that form
metabolic and gene regulatory networks. Both models include
intracellular and extracellular metabolites, diffusion, transport, cell
division, death, and genome evolution. The main differences are that
Evo²Sim uses a coarse-grained genome with explicit promoters and
gene-expression regulation, whereas MagicSoup uses nucleotide-level
coding regions and protein domains; conversely, MagicSoup features more
detailed reaction kinetics, thermodynamic constraints, and energetic
coupling between protein domains.

**Reference:** Rocabert, C., Knibbe, C., Consuegra, J., Schneider, D., &
Beslon, G. (2017). Beware batch culture: Seasonality and niche
construction predicted to favor bacterial adaptive diversification.
*PLoS Computational Biology*, 13(3), e1005459.
https://doi.org/10.1371/journal.pcbi.1005459

### Evolution of Metabolic Networks (Flamm et al., 2010)

[Scientific publication and model
description](https://link.springer.com/article/10.1186/1759-2208-1-4)

This framework is particularly interesting because it explicitly models
the evolution of metabolic networks from genome-encoded catalytic
molecules. Like MagicSoup, it combines genetics, chemical reactions,
enzyme kinetics, thermodynamics, and evolutionary selection. However, it
focuses on the early evolution of metabolism and an RNA-world scenario,
using an artificial chemistry based on graph rewriting and ribozymes
rather than protein domains and conventional biochemical reactions. It
also lacks MagicSoup's explicit spatial environment, cell migration,
diffusion, and cell-cell interactions.

**Reference:** Flamm, C., Ullrich, A., Ekker, H., Mann, M., Högerl, D.,
Rohrschneider, M., Sauer, S., Scheuermann, G., Klemm, K., Hofacker, I.
L., & Stadler, P. F. (2010). Evolution of metabolic networks: A
computational framework. *Journal of Systems Chemistry*, 1, 4.
https://doi.org/10.1186/1759-2208-1-4

### Evolution of Complex Modular Biological Networks (Hintze & Adami, 2008)

[Scientific
publication](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0040023)

This simulation evolves artificial cells containing genomes, proteins,
and metabolites within an artificial chemistry. Similar to MagicSoup,
mutations change the encoded proteins and thereby alter metabolic
networks, which evolve under environmental selection. The authors
specifically investigate the emergence of network modularity,
robustness, and functional organization. The main difference is that
their artificial chemistry and metabolic processes are considerably more
abstract than MagicSoup's biochemical model, which explicitly
incorporates reaction kinetics, energy constraints, transport, and
regulatory domains.

**Reference:** Hintze, A., & Adami, C. (2008). Evolution of complex
modular biological networks. *PLoS Computational Biology*, 4(2), e23.
https://doi.org/10.1371/journal.pcbi.0040023

### The Evolution of Connectivity in Metabolic Networks (Pfeiffer et al., 2005)

[Scientific
publication](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0030228)

This work investigates how metabolic networks evolve from initially
unspecific enzymes into networks with specialized enzymes and highly
connected metabolites. It shares several aspects with MagicSoup,
including intracellular and extracellular metabolites, transporters,
enzyme kinetics, gene duplication, and selection based on metabolic
performance. However, it uses a predefined biochemical reaction system
and evolves enzyme specificities and gene copy numbers rather than
nucleotide-level genomes and their encoded proteins. It also lacks
MagicSoup's spatial dynamics and regulatory networks.

**Reference:** Pfeiffer, T., Soyer, O. S., & Bonhoeffer, S. (2005). The
evolution of connectivity in metabolic networks. *PLoS Biology*, 3(7),
e228. https://doi.org/10.1371/journal.pbio.0030228

### Simulating the Evolution of Signal Transduction Pathways (Soyer et al., 2006)

[Scientific publication](https://doi.org/10.1016/j.jtbi.2005.11.024)

This work focuses on the evolution of protein interaction networks
responsible for signal transduction. Like MagicSoup, it uses genetic
variation and selection to evolve complex pathways consisting of
interacting proteins, with the goal of understanding how functional
networks emerge. In particular, it investigates the evolution of
networks capable of producing chemotactic behavior. Unlike MagicSoup, it
focuses on regulatory interactions rather than combining metabolic and
regulatory networks, and does not model explicit biochemical reaction
chemistry or spatially distributed cells.

**Reference:** Soyer, O. S., Pfeiffer, T., & Bonhoeffer, S. (2006).
Simulating the evolution of signal transduction pathways. *Journal of
Theoretical Biology*, 241(2), 223--232.
https://doi.org/10.1016/j.jtbi.2005.11.024




