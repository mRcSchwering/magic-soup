# Other Simulations

### MCell and CellBlender

[MCell](https://mcell.org/) and [CellBlender](https://github.com/mcellteam/cellblender).
MCell is an agent-based reaction-diffusion program that simulates the movement and interaction of molecules in 3D space ([documentation](https://mcell.org/mcell4_documentation/sim_env_overview.html)).
CellBlender is a Blender add-on that can be used as visual interface for MCell.
With it you can create beautiful animations of your experiment -
_e.g._ of a oscillating Min system as done in [[1]](https://www.pnas.org/doi/abs/10.1073/pnas.0505825102).

[[1]](https://www.pnas.org/doi/abs/10.1073/pnas.0505825102) Rex, Kerr., Herbert, Levine., Terrence, J., Sejnowski., Wouter-Jan, Rappel. (2006). Division accuracy in a stochastic model of Min oscillations in Escherichia coli. Proceedings of the National Academy of Sciences of the United States of America, 103(2):347-352. doi: 10.1073/PNAS.0505825102

### VCell

[VCell](https://vcell.org/) is a comprehensive platform for modeling cell biological systems that is built on a central database and disseminated as a web application.
It has many features. For example one can create a 3D cell based on microscopy images, then add reactions and pathways, and let VCell setup the simulation.
[[2]](https://pubmed.ncbi.nlm.nih.gov/22482950/) describes spatial modelling with VCell.

[[2]](https://pubmed.ncbi.nlm.nih.gov/22482950/) Anne, E., Cowan., Ion, I., Moraru., James, C., Schaff., Boris, M., Slepchenko., Leslie, M., Loew. (2012). Spatial modeling of cell signaling networks.. Methods in Cell Biology, 110:195-221. doi: 10.1016/B978-0-12-388403-9.00008-4


### CellSys/TiSim

[CellSys/TiSim](https://www.hoehme.com/software/tisim) is a modular software tool for efficient off-lattice simulation of growth and organization processes in multicellular systems in two and three dimensions. It implements an agent-based model that approximates cells as isotropic, elastic and adhesive objects. Cell migration is modeled by an equation of motion for each cell. (from [[3]](https://academic.oup.com/bioinformatics/article/26/20/2641/193478))

[[3]](https://academic.oup.com/bioinformatics/article/26/20/2641/193478) Stefan, Hoehme., Dirk, Drasdo. (2010). A cell-based simulation software for multi-cellular systems. Bioinformatics, 26(20):2641-2642. doi: 10.1093/BIOINFORMATICS/BTQ437


### Genie

[Genie](https://cartwrig.ht/apps/genie/).
Cool simulation to visualize genetic drift and gene flow in a population.
It's browser-based, you can try it out on their website.
The paper is [[4]](https://evolution-outreach.biomedcentral.com/articles/10.1186/s12052-022-00161-7).

[[4]](https://evolution-outreach.biomedcentral.com/articles/10.1186/s12052-022-00161-7) Castillo, A.I., Roos, B.H., Rosenberg, M.S. et al. Genie: an interactive real-time simulation for teaching genetic drift. Evo Edu Outreach 15, 3 (2022). doi.org: 10.1186/s12052-022-00161-7


### MagicSoup

_MagicSoup_ is an agent-based simulation with cells living, interacting, and replicating on a 2D grid.
In this regard it is similar to _Genie_.
In contrast to _Genie_ however, it includes an algorithm which translates genomes into working metabolic and transduction pathways.
In principle these pathways may seem similar to the ones in _VCell_.
But _VCell_ uses more sophisticated algorithms to model these pathways.
Additionally, in _VCell_ the cell itself can have a 2D or 3D shape.
In _MCell_ molecule movements within the 3D cell can even be simulated spatio-temporally.
This doesn't exist in _MagicSoup_ for now.
Here, a cell is basically a homogenous bag of molecules.
So, something like a Min system will not work in _MagicSoup_.







