import random
from itertools import product
import math
from io import BytesIO
import pickle
from pathlib import Path
import torch
from magicsoup.constants import CODON_SIZE
from magicsoup.util import moore_nghbrhd, round_down, random_genome, randstr
from magicsoup.containers import (
    Cell,
    Chemistry,
    ProteinFact,
    CatalyticDomainFact,
    TransporterDomainFact,
    RegulatoryDomainFact,
)
from magicsoup.kinetics import Kinetics
from magicsoup.genetics import Genetics


def _torch_load(map_loc: str | None = None):
    # Closure rather than a lambda to preserve map_loc
    return lambda b: torch.load(BytesIO(b), map_location=map_loc)


class _CPU_Unpickler(pickle.Unpickler):
    """Inject map_location when unpickling tensor objects"""

    def __init__(self, *args, map_location: str | None = None, **kwargs):
        self._map_location = map_location
        super().__init__(*args, **kwargs)

    def find_class(self, module: str, name: str):
        if module == "torch.storage" and name == "_load_from_bytes":
            return _torch_load(map_loc=self._map_location)
        else:
            return super().find_class(module, name)


class World:
    """
    This is the main object for running the simulation.
    It holds all information and includes all methods for advancing the simulation.

    Parameters:
        chemistry: The chemistry object that defines molecule species and reactions for this simulation.
        map_size: Size of world map as number of pixels in x- and y-direction.
        abs_temp: Absolute temperature in Kelvin will influence the free Gibbs energy calculation of reactions.
            Higher temperature will give the reaction quotient term higher importance.
        mol_map_init: How to initialize molecule maps (`randn` or `zeros`).
            `randn` is normally distributed N(10, 1), `zeros` is all zeros.
        start_codons: start codons which start a coding sequence (translation only happens within coding sequences).
        stop_codons: stop codons which stop a coding sequence (translation only happens within coding sequences).
        device: Device to use for tensors (see [pytorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)).
            This can be used to move most calculations to a GPU.
        workers: Number of multiprocessing workers to use.
            These are used to parallelize some calculations that can only be done on the CPU.

    Most attributes on this class describe the current state of molecules and cells.
    Whenever molecules are listed or represented in one dimension, they are ordered the same way as in `chemistry.molecules`.
    Likewise, cells are always ordered the same way as in `world.genomes` (see below).
    The index of a certain cell is the index of that cell in `world.genomes`.
    It is the same index as `cell.idx` of a cell object you retrieved with `world.get_cell()`.
    But whenever an operation modifies the number of cells (like `world.kill_cells()` or `world.divide_cells()`),
    cells get new indexes. Here are the most important attributes:

    Attributes:
        genomes: A list of cell genomes. Each cell's index in this list is what is referred to as the cell index.
            The cell index is used for the same cell in other orderings of cells (_e.g._ `labels`, `cell_divisions`, `cell_molecules`).
        labels: List of cell labels. Cells are ordered as in `world.genomes`. Labels are strings that can be used to
            track cell origins. When adding new cells (`world.add_cells()`) a random label is assigned to each cell.
            If a cell divides, its descendants will have the same label.
        cell_map: Boolean 2D tensor referencing which pixels are occupied by a cell.
            Dimension 0 represents the x, dimension 1 y.
        molecule_map: Float 3D tensor describing how many molecules (in mol) of each molecule species exist on every pixel in this world.
            Dimension 0 describes the molecule species. They are in the same order as `chemistry.molecules`.
            Dimension 1 represents x, dimension 2 y.
            So, `world.molecule_map[0, 1, 2]` is number of molecules of the 0th molecule species on pixel 1, 2.
        cell_molecules: Float 2D tensor describing the number of molecules (in mol) for each molecule species in each cell.
            Dimension 0 is the cell index. It is the same as in `world.genomes` and the same as on a cell object (`cell.idx`).
            Dimension 1 describes the molecule species. They are in the same order as `chemistry.molecules`.
            So, `world.cell_molecules[0, 1]` represents how many mol of the 1st molecule species the 0th cell contains.
        cell_survival: Integer 1D tensor describing how many time steps each cell survived.
            This tensor is for monitoring and doesn't have any other effect.
            Cells are in the same as in `world.genomes` and the same as on a cell object (`cell.idx`).
        cell_divisions: Integer 1D tensor describing how many times each cell's ancestors divided.
            This tensor is for monitoring and doesn't have any other effect.
            Cells are in the same order as in `world.genomes` and the same as on a cell object (`cell.idx`).

    Methods for advancing the simulation and to use during a simulation:

    - [add_cells()][magicsoup.world.World.add_cells] add new cells and place them randomly on the map
    - [divide_cells()][magicsoup.world.World.divide_cells] replicate existing cells
    - [update_cells()][magicsoup.world.World.update_cells] update existing cells if their genome has changed
    - [kill_cells()][magicsoup.world.World.kill_cells] kill existing cells
    - [move_cells()][magicsoup.world.World.move_cells] move existing cells to a random position in their Moore's neighborhood
    - [diffuse_molecules()][magicsoup.world.World.diffuse_molecules] let molecules diffuse and permeate by one time step
    - [degrade_molecules()][magicsoup.world.World.degrade_molecules] let molecules degrade by one time step
    - [increment_cell_survival()][magicsoup.world.World.increment_cell_survival] increment `world.cell_survival` by 1
    - [enzymatic_activity()][magicsoup.world.World.enzymatic_activity] let cell proteins work for one time step

    If you want to get a cell with all information about its contents and its current environment use [get_cell()][magicsoup.world.World.get_cell].
    During the simulation you should however work directly with the tensors mentioned above for performance reasons.

    Furthermore, there are methods for saving and loading a simulation.
    For any new simulation use [save()][magicsoup.world.World.save] once to save the whole world object (with chemistry, genetics, kinetics)
    to a pickle file. You can restore it with [from_file()][magicsoup.world.World.from_file] later on.
    Then, to save the world's state you can use [save_state()][magicsoup.world.World.save_state].
    This is a quick, lightweight save, but it only saves things that change during the simulation.
    Use [load_state()][magicsoup.world.World.load_state] to re-load a certain state.

    Finally, the `world` object carries `world.genetics`, `world.kinetics`, and `world.chemistry`
    (which is just a reference to the chemistry object that was used when initializing `world`).
    Usually, you don't need to touch them.
    But, if you want to override them or look into some details, see the docstrings of their classes for more information.
    """

    def __init__(
        self,
        chemistry: Chemistry,
        map_size: int = 128,
        abs_temp: float = 310.0,
        mol_map_init: str = "randn",
        start_codons: tuple[str, ...] = ("TTG", "GTG", "ATG"),
        stop_codons: tuple[str, ...] = ("TGA", "TAG", "TAA"),
        device: str = "cpu",
        workers: int = 2,
    ):
        if not torch.cuda.is_available():
            device = "cpu"

        self.device = device
        self.workers = workers
        self.map_size = map_size
        self.abs_temp = abs_temp
        self.chemistry = chemistry

        self.genetics = Genetics(
            start_codons=start_codons, stop_codons=stop_codons, workers=workers
        )

        self.kinetics = Kinetics(
            molecules=chemistry.molecules,
            reactions=chemistry.reactions,
            abs_temp=abs_temp,
            device=self.device,
            scalar_enc_size=max(self.genetics.one_codon_map.values()),
            vector_enc_size=max(self.genetics.two_codon_map.values()),
        )

        mol_degrads: list[float] = []
        diffusion: list[torch.nn.Conv2d] = []
        permeation: list[float] = []
        for mol in chemistry.molecules:
            mol_degrads.append(math.exp(-math.log(2) / mol.half_life))
            diffusion.append(self._get_diffuse(mol_diff_rate=mol.diffusivity))
            permeation.append(self._get_permeate(mol_perm_rate=mol.permeability))

        self.n_molecules = len(chemistry.molecules)
        self._int_mol_idxs = list(range(self.n_molecules))
        self._ext_mol_idxs = list(range(self.n_molecules, self.n_molecules * 2))
        self._mol_degrads = mol_degrads
        self._diffusion = diffusion
        self._permeation = permeation

        self._nghbrhd_map = {
            (x, y): torch.tensor(moore_nghbrhd(x, y, map_size)).to(self.device)
            for x, y in product(range(map_size), range(map_size))
        }

        self.n_cells = 0
        self.genomes: list[str] = []
        self.labels: list[str] = []
        self.cell_map: torch.Tensor = torch.zeros(map_size, map_size).to(device).bool()
        self.cell_positions: torch.Tensor = torch.zeros(0, 2).to(device).long()
        self.cell_survival: torch.Tensor = torch.zeros(0).to(device).int()
        self.cell_divisions: torch.Tensor = torch.zeros(0).to(device).int()
        self.cell_molecules: torch.Tensor = torch.zeros(0, self.n_molecules).to(device)
        self.molecule_map: torch.Tensor = self._get_molecule_map(
            n=self.n_molecules, size=map_size, init=mol_map_init
        )

    def get_cell(
        self,
        by_idx: int | None = None,
        by_position: tuple[int, int] | None = None,
    ) -> Cell:
        """
        Get a cell with information about its current environment.
        Raises `ValueError` if cell was not found.

        Parameters:
            by_idx: get cell by cell index (`cell.idx`)
            by_position: get cell by position (x, y)

        Returns:
            The searched cell.

        For performance reasons most cell attributes are maintained in tensors during the simulation.
        When you call `world.get_cell()` all information about a cell is gathered in one object.
        This is a convenience function for interactive use.
        It's not very performant.
        """
        idx = -1
        if by_idx is not None:
            idx = by_idx
        if by_position is not None:
            pos = torch.tensor(by_position)
            mask = (self.cell_positions == pos).all(dim=1)
            idxs = torch.argwhere(mask).flatten().tolist()
            if len(idxs) == 0:
                raise ValueError(f"Cell at {by_position} not found")
            idx = idxs[0]

        genome = self.genomes[idx]
        (cdss,) = self.genetics.translate_genomes(genomes=[genome])
        pos = self.cell_positions[idx]

        return Cell(
            idx=idx,
            genome=genome,
            proteome=self.kinetics.get_proteome(proteome=cdss),
            position=tuple(pos.tolist()),  # type: ignore
            int_molecules=self.cell_molecules[idx, :],
            ext_molecules=self.molecule_map[:, pos[0], pos[1]],
            label=self.labels[idx],
            n_survived_steps=int(self.cell_survival[idx].item()),
            n_divisions=int(self.cell_divisions[idx].item()),
        )

    def get_neighbors(self, cell_idxs: list[int]) -> list[tuple[int, int]]:
        """
        For a list of cells get pairs of cell indexes of cells which are (Moore's) neighbors.
        Each pair of neighboring cells is only returned once.

        Parameters:
            cell_idxs: Indexes of cells for which to find neighbors

        Returns:
            List of tuples of cell indexes of 2 unique neighboring cells each.
        """

        if len(cell_idxs) == 0:
            return []

        # rm duplicates to avoid unnecessary compute
        cell_idxs = list(set(cell_idxs))

        nghbrs: list[tuple[int, int]] = []
        for c_idx in cell_idxs:
            c_pos = self.cell_positions[c_idx]
            nghbrhd = self._nghbrhd_map[tuple(c_pos.tolist())]  # type: ignore
            pxls = nghbrhd[self.cell_map[nghbrhd[:, 0], nghbrhd[:, 1]]]
            n = pxls.size(0)

            if n == 0:
                continue

            idx_ts = [(self.cell_positions == d).all(dim=1).argwhere() for d in pxls]
            n_idxs: list[int] = torch.cat(idx_ts).flatten().tolist()
            nghbrs.extend(tuple(sorted([c_idx, d])) for d in n_idxs)  # type:ignore

        return list(set(nghbrs))

    def generate_genome(self, proteome: list[ProteinFact], size: int = 500) -> str:
        """
        Generate a random genome that encodes a desired proteome

        Arguments:
            proteome: list of [ProteinFactories][magicsoup.containers.ProteinFact] that describe each protein
            size: total length of the resulting genome (in nucleotides/base pairs)

        Returns:
            Genome as string of nucleotide letters

        This function uses the mappings from [Genetics][magicsoup.genetics.Genetics]
        and [Kinetics][magicsoup.kinetics.Kinetics] in reverse to generate a genome.
        So, the same [World][magicsoup.world.World] object should be used when running a simulation
        with these generated genomes.
        Some domain specifications are given in `proteome`, some (such as the kinetics parameters)
        will be assigned randomly.

        Note, this function is neither particularly intelligent nor performant.
        It can happen that some proteins of the desired proteome are missing.
        This can happen for example when a stop codon appear accidentally inside a domain definition,
        terminating this protein pre-maturely.
        At the same time, the resulting genome can also include proteins that were not defined.
        E.g. the reverse-complement can encode additional proteins.
        This function tries to avoid this, but is not capable of avoiding it all the time.
        """
        all_mols = self.chemistry.molecules
        all_reacts = self.chemistry.reactions
        dom_size = self.genetics.dom_size

        req_nts = 0
        reacts = [(tuple(sorted(s)), tuple(sorted(p))) for s, p in all_reacts]
        fwd_bwd_reacts = reacts + [(p, s) for s, p in reacts]

        for pi, prot in enumerate(proteome):
            req_nts += 2 * CODON_SIZE + dom_size * prot.n_domains
            for dom in prot.domain_facts:
                if isinstance(dom, TransporterDomainFact):
                    if dom.molecule not in all_mols:
                        raise ValueError(
                            f"ProteinFact {pi} has this molecule defined: {dom.molecule.name}."
                            " This world's chemistry doesn't define this molecule species."
                        )
                if isinstance(dom, RegulatoryDomainFact):
                    if dom.effector not in all_mols:
                        raise ValueError(
                            f"ProteinFact {pi} has this effector defined: {dom.effector.name}."
                            " This world's chemistry doesn't define this molecule species."
                        )
                if isinstance(dom, CatalyticDomainFact):
                    subs = sorted(dom.substrates)
                    prods = sorted(dom.products)
                    react = (tuple(subs), tuple(prods))
                    if react not in fwd_bwd_reacts:
                        reactstr = f"{' + '.join(d.name for d in subs)} <-> {' + '.join(d.name for d in prods)}"
                        raise ValueError(
                            f"ProteinFact {pi} has this reaction defined: {reactstr}."
                            " This world's chemistry doesn't define this reaction."
                        )

        if req_nts > size:
            raise ValueError(
                "Genome size too small."
                f" The given proteome would require at least {req_nts} nucleotides."
                f" But the given genome size is size={size}."
            )

        cdss = self._get_genome_sequences(proteome=proteome)
        n_p_pads = len(cdss) + 1
        n_d_pads = sum(len(d) + 1 for d in cdss)

        n_pad_nts = size - req_nts
        pad_size = n_pad_nts / (n_p_pads * 0.7 + n_d_pads * 0.3)
        d_pad_size = round_down(pad_size * 0.3, to=3)
        d_pad_total = n_d_pads * d_pad_size
        p_pad_size = round_down((n_pad_nts - d_pad_total) / n_p_pads, to=1)
        p_pad_total = n_p_pads * p_pad_size
        remaining_nts = n_pad_nts - d_pad_total - p_pad_total

        start_codons = self.genetics.start_codons
        stop_codons = self.genetics.stop_codons
        excl_cdss = start_codons + stop_codons
        excl_doms = excl_cdss + list(self.genetics.domain_map)
        p_pads = [random_genome(s=p_pad_size, excl=excl_cdss) for _ in range(n_p_pads)]
        d_pads = [random_genome(s=d_pad_size, excl=excl_doms) for _ in range(n_d_pads)]
        tail = random_genome(s=remaining_nts, excl=excl_cdss)

        parts: list[str] = []
        for cds in cdss:
            parts.append(p_pads.pop())
            parts.append(random.choice(start_codons))
            for dom_seq in cds:
                parts.append(d_pads.pop())
                parts.append(dom_seq)
            parts.append(d_pads.pop())
            parts.append(random.choice(stop_codons))
        parts.append(p_pads.pop())
        parts.append(tail)

        return "".join(parts)

    def add_cells(self, genomes: list[str]) -> list[int]:
        """
        Create new cells and place them randomly on the map.
        All lists and tensors that reference cells will be updated.

        Parameters:
            genomes: List of genomes of the newly added cells

        Returns:
            The indexes of successfully added cells.

        Each cell will be placed randomly on the map and receive half the molecules of the pixel where it was added.
        If there are less pixels left on the cell map than cells you want to add,
        only the remaining pixels will be filled with new cells.
        Each cell will also receive a random label.
        """
        genomes = [d for d in genomes if len(d) > 0]
        n_new_cells = len(genomes)
        if n_new_cells == 0:
            return []

        free_pos = self._find_free_random_positions(n_cells=n_new_cells)
        n_avail_pos = free_pos.size(0)
        if n_avail_pos == 0:
            return []

        if n_avail_pos < n_new_cells:
            n_new_cells = n_avail_pos
            random.shuffle(genomes)
            genomes = genomes[:n_new_cells]

        proteomes = self.genetics.translate_genomes(genomes=genomes)

        new_genomes = []
        new_proteomes = []
        new_labels = []
        for proteome, genome in zip(proteomes, genomes):
            if len(proteome) > 0:
                new_genomes.append(genome)
                new_proteomes.append(proteome)
                new_labels.append(randstr(n=12))

        n_new_cells = len(new_genomes)
        if n_new_cells == 0:
            return []

        new_pos = free_pos[:n_new_cells]
        new_idxs = list(range(self.n_cells, self.n_cells + n_new_cells))
        self.n_cells += n_new_cells
        self.genomes.extend(new_genomes)
        self.labels.extend(new_labels)

        self.cell_survival = self._expand_c(t=self.cell_survival, n=n_new_cells)
        self.cell_positions = self._expand_c(t=self.cell_positions, n=n_new_cells)
        self.cell_divisions = self._expand_c(t=self.cell_divisions, n=n_new_cells)
        self.cell_molecules = self._expand_c(t=self.cell_molecules, n=n_new_cells)

        n_max_prots = max(len(d) for d in new_proteomes)
        self.kinetics.increase_max_proteins(max_n=n_max_prots)
        self.kinetics.increase_max_cells(by_n=n_new_cells)
        self.kinetics.set_cell_params(cell_idxs=new_idxs, proteomes=new_proteomes)

        # occupy positions
        xs = new_pos[:, 0]
        ys = new_pos[:, 1]
        self.cell_map[xs, ys] = True
        self.cell_positions[new_idxs] = new_pos

        # cell is picking up half the molecules of the pxl it is born on
        pickup = self.molecule_map[:, xs, ys] * 0.5
        self.cell_molecules[new_idxs, :] += pickup.T
        self.molecule_map[:, xs, ys] -= pickup

        return new_idxs

    def divide_cells(self, cell_idxs: list[int]) -> list[tuple[int, int]]:
        """
        Bring cells to divide.
        All lists and tensors that reference cells will be updated.

        Parameters:
            cell_idxs: Cell indexes of the cells that should divide.

        Returns:
            A list of tuples of descendant cell indexes.

        Each cell divides by creating a clone of itself on a random pixel next to itself (Moore's neighborhood).
        If every pixel in its Moore's neighborhood is taken, it will not be able to divide.
        Both, the original ancestor cell and the newly placed cell will become the descendants.
        They will have the same genome, proteome, and label.
        Both descendants share all molecules equally.
        So each descendant cell will get half the molecules of the ancestor cell.

        Both descendants will share the same number of divisions.
        It is incremented by 1 for both cells.
        So, _e.g._ if a cell with `n_divisions=2` divides, its descendants both have `n_divisions=3`.
        For both of them the number of survived steps is 0 after the division.

        Of the list of descendant index tuples,
        the first descendant in each tuple is the cell that still lives on the same pixel.
        The second descendant in that tuple is the cell that was newly created.
        """
        if len(cell_idxs) == 0:
            return []

        # duplicates could lead to unexpected results
        cell_idxs = list(set(cell_idxs))

        (
            parent_idxs,
            child_idxs,
            child_pos,
        ) = self._divide_cells_as_possible(parent_idxs=cell_idxs)

        n_new_cells = len(child_idxs)
        if n_new_cells == 0:
            return []

        self.n_cells += n_new_cells

        self.cell_survival = self._expand_c(t=self.cell_survival, n=n_new_cells)
        self.cell_positions = self._expand_c(t=self.cell_positions, n=n_new_cells)
        self.cell_divisions = self._expand_c(t=self.cell_divisions, n=n_new_cells)
        self.cell_molecules = self._expand_c(t=self.cell_molecules, n=n_new_cells)
        self.kinetics.increase_max_cells(by_n=n_new_cells)
        self.kinetics.copy_cell_params(from_idxs=parent_idxs, to_idxs=child_idxs)

        # position new cells
        # cell_map was already set in loop before
        self.cell_positions[child_idxs] = torch.stack(child_pos)

        # cells share molecules and increment cell divisions
        descendant_idxs = parent_idxs + child_idxs
        self.cell_molecules[child_idxs] = self.cell_molecules[parent_idxs]
        self.cell_molecules[descendant_idxs] *= 0.5
        self.cell_divisions[child_idxs] = self.cell_divisions[parent_idxs]
        self.cell_divisions[descendant_idxs] += 1
        self.cell_survival[descendant_idxs] = 0

        return list(zip(parent_idxs, child_idxs))

    def update_cells(self, genome_idx_pairs: list[tuple[str, int]]):
        """
        Update existing cells with new genomes.

        Parameters:
            genome_idx_pairs: List of tuples of genomes and cell indexes

        The indexes refer to the index of each cell that is changed.
        The genomes refer to the genome of each cell that is changed.
        """
        if len(genome_idx_pairs) == 0:
            return

        kill_idxs = []
        transl_idxs = []
        transl_genomes = []
        for genome, idx in genome_idx_pairs:
            if len(genome) > 0:
                transl_genomes.append(genome)
                transl_idxs.append(idx)
            else:
                kill_idxs.append(idx)

        proteomes = self.genetics.translate_genomes(genomes=transl_genomes)

        set_idxs = []
        set_proteomes = []
        for proteome, idx, genome in zip(proteomes, transl_idxs, transl_genomes):
            if len(proteome) > 0:
                set_proteomes.append(proteome)
                set_idxs.append(idx)
                self.genomes[idx] = genome
            else:
                kill_idxs.append(idx)

        if len(set_proteomes) > 0:
            max_prots = max(len(d) for d in set_proteomes)
            self.kinetics.increase_max_proteins(max_n=max_prots)
            self.kinetics.set_cell_params(cell_idxs=set_idxs, proteomes=set_proteomes)

        self.kill_cells(cell_idxs=kill_idxs)

    def kill_cells(self, cell_idxs: list[int]):
        """
        Remove existing cells.
        All lists and tensors referencing cells will be updated.

        Parameters:
            cell_idxs: Indexes of the cells that should die

        Cells that are killed dump their molecule contents onto the pixel they used to live on.

        Cells will be removed from all lists and tensors that reference cells.
        Thus, after killing cells the index of some living cells will be updated.
        E.g. if there are 10 cells and you kill the cell with index 8 (the 9th cell),
        the cell that used to have index 9 (10th cell) will now have index 9.
        """
        n_killed_cells = len(cell_idxs)
        if n_killed_cells == 0:
            return

        # duplicates could raise error later
        cell_idxs = list(set(cell_idxs))

        xs = self.cell_positions[cell_idxs, 0]
        ys = self.cell_positions[cell_idxs, 1]
        self.cell_map[xs, ys] = False

        spillout = self.cell_molecules[cell_idxs, :]
        self.molecule_map[:, xs, ys] += spillout.T

        n_cells = self.cell_survival.size(0)
        keep = torch.ones(n_cells, dtype=torch.bool).to(self.device)
        keep[cell_idxs] = False
        self.cell_survival = self.cell_survival[keep]
        self.cell_positions = self.cell_positions[keep]
        self.cell_divisions = self.cell_divisions[keep]
        self.cell_molecules = self.cell_molecules[keep]
        self.kinetics.remove_cell_params(keep=keep)

        for idx in sorted(cell_idxs, reverse=True):
            self.genomes.pop(idx)
            self.labels.pop(idx)

        self.n_cells -= n_killed_cells

    def move_cells(self, cell_idxs: list[int]):
        """
        Move cells to a random position in their Moore's neighborhood.
        If every pixel in the cells' Moore neighborhood is taken the cell will not be moved.
        `world.cell_map` will be updated.

        Parameters:
            cell_idxs: Indexes of cells that should be moved
        """
        if len(cell_idxs) == 0:
            return

        # duplicates could lead to unexpected results
        cell_idxs = list(set(cell_idxs))

        for cell_idx in cell_idxs:
            old_pos = self.cell_positions[cell_idx]
            nghbrhd = self._nghbrhd_map[tuple(old_pos.tolist())]  # type: ignore
            pxls = nghbrhd[~self.cell_map[nghbrhd[:, 0], nghbrhd[:, 1]]]
            n = pxls.size(0)

            if n == 0:
                continue

            # move cells
            new_pos = pxls[random.randint(0, n - 1)]
            self.cell_map[old_pos[0], old_pos[1]] = False
            self.cell_map[new_pos[0], new_pos[1]] = True
            self.cell_positions[cell_idx] = new_pos

    def reposition_cells(self, cell_idxs: list[int]):
        """
        Reposition cells randomly on cell map without changing them.

        Parameters:
            cell_idxs: Indexes of cells that should be moved

        Cells are removed from their current pixels on the cell map
        and repositioned randomly on any free pixel on the cell map.
        Only cell positions change (_e.g._ genomes, proteomes, cell molecules, cell indexes stay the same).
        """
        n_cells = len(cell_idxs)
        if n_cells == 0:
            return

        # duplicates could lead to unexpected results
        cell_idxs = list(set(cell_idxs))

        old_xs = self.cell_positions[cell_idxs, 0]
        old_ys = self.cell_positions[cell_idxs, 1]
        self.cell_map[old_xs, old_ys] = False

        new_pos = self._find_free_random_positions(n_cells=n_cells)
        new_xs = new_pos[:, 0]
        new_ys = new_pos[:, 1]

        self.cell_map[new_xs, new_ys] = True
        self.cell_positions[cell_idxs] = new_pos

    def enzymatic_activity(self):
        """
        Catalyze reactions for one time step.
        This includes molecule transport into or out of the cell.
        `world.molecule_map` and `world.cell_molecules` are updated.
        """
        if self.n_cells == 0:
            return

        xs = self.cell_positions[:, 0]
        ys = self.cell_positions[:, 1]
        X = torch.cat([self.cell_molecules, self.molecule_map[:, xs, ys].T], dim=1)
        X = self.kinetics.integrate_signals(X=X)

        self.molecule_map[:, xs, ys] = X[:, self._ext_mol_idxs].T
        self.cell_molecules = X[:, self._int_mol_idxs]

    @torch.no_grad()
    def diffuse_molecules(self):
        """
        Let molecules in molecule map diffuse and permeate through cell membranes
        by one time step.
        `world.molecule_map` and `world.cell_molecules` are updated.

        By how much each molcule species diffuses around the world map and permeates
        into or out of cells if defined on the `Molecule` objects itself.
        See `Molecule` for more information.
        """
        n_pxls = self.map_size**2
        for mol_i, diffuse in enumerate(self._diffusion):
            total_before = self.molecule_map[mol_i].sum()
            before = self.molecule_map[mol_i].unsqueeze(0).unsqueeze(1)
            after = diffuse(before)
            self.molecule_map[mol_i] = torch.squeeze(after, 0).squeeze(0)
            total_after = self.molecule_map[mol_i].sum()

            # attempt to fix the problem that convolusion makes a small amount of
            # molecules appear or disappear (I think because floating point)
            self.molecule_map[mol_i] += (total_before - total_after) / n_pxls
            self.molecule_map[mol_i] = self.molecule_map[mol_i].clamp(0.0)

        if self.n_cells == 0:
            return

        xs = self.cell_positions[:, 0]
        ys = self.cell_positions[:, 1]
        X = torch.cat([self.cell_molecules, self.molecule_map[:, xs, ys].T], dim=1)

        for mol_i, permeate in enumerate(self._permeation):
            d_int = X[:, mol_i] * permeate
            d_ext = X[:, mol_i + self.n_molecules] * permeate
            X[:, mol_i] += d_ext - d_int
            X[:, mol_i + self.n_molecules] += d_int - d_ext

        self.molecule_map[:, xs, ys] = X[:, self._ext_mol_idxs].T
        self.cell_molecules = X[:, self._int_mol_idxs]

    def degrade_molecules(self):
        """
        Degrade molecules in world map and cells by one time step.
        `world.molecule_map` and `world.cell_molecules` are updated.

        How quickly each molecule species degrades depends on its half life
        which is defined on the `Molecule` object of that species.
        """
        for mol_i, degrad in enumerate(self._mol_degrads):
            self.molecule_map[mol_i] *= degrad
            self.cell_molecules[:, mol_i] *= degrad

    def increment_cell_survival(self):
        """
        Increment `world.cell_survival` by 1.
        This is for monitoring and doesn't have any other effect.
        """
        self.cell_survival += 1

    def save(self, rundir: Path, name: str = "world.pkl"):
        """
        Write whole world object to pickle file

        Parameters:
            rundir: Directory of the pickle file
            name: Name of the pickle file

        This is a big and slow save.
        It saves everything needed to restore the world with its chemistry and genetics.
        Use [from_file()][magicsoup.world.World.from_file] to restore it.
        For a small and quick save use [save_state()][magicsoup.world.World.save_state].
        """
        rundir.mkdir(parents=True, exist_ok=True)
        with open(rundir / name, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def from_file(
        self,
        rundir: Path,
        name: str = "world.pkl",
        device: str | None = None,
        workers: int | None = None,
    ) -> "World":
        """
        Restore previously saved world from pickle file.
        The file had to be saved with [save()][magicsoup.world.World.save].

        Parameters:
            rundir: Directory of the pickle file
            name: Name of the pickle file
            device: Optionally set device to which tensors should be loaded.
                Default is the device they had when they were saved.

        Returns:
            A new `world` instance.
        """
        with open(rundir / name, "rb") as fh:
            unpickler = _CPU_Unpickler(fh, map_location=device)
            obj: "World" = unpickler.load()

        if device is not None:
            obj.device = device
            obj.kinetics.device = device

        if workers is not None:
            obj.workers = workers
            obj.genetics.workers = workers

        return obj

    def save_state(self, statedir: Path):
        """
        Save current state only

        Parameters:
            statedir: Directory to store files in (there are multiple files per state)

        This is a small and quick save.
        It only saves things which change during the simulation.
        Restore a certain state with [load_state()][magicsoup.world.World.load_state].

        To restore a whole `world` object you need to save it at least once with [save()][magicsoup.world.World.save].
        Then, `world.save_state()` can be used to save different states of that world object.
        """
        statedir.mkdir(parents=True, exist_ok=True)
        torch.save(self.cell_molecules, statedir / "cell_molecules.pt")
        torch.save(self.cell_map, statedir / "cell_map.pt")
        torch.save(self.molecule_map, statedir / "molecule_map.pt")
        torch.save(self.cell_survival, statedir / "cell_survival.pt")
        torch.save(self.cell_positions, statedir / "cell_positions.pt")
        torch.save(self.cell_divisions, statedir / "cell_divisions.pt")

        lines: list[str] = []
        for idx, (genome, label) in enumerate(zip(self.genomes, self.labels)):
            lines.append(f">{idx} {label}\n{genome}")

        with open(statedir / "cells.fasta", "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    def load_state(
        self,
        statedir: Path,
        ignore_cell_params: bool = False,
    ):
        """
        Load a saved world state.
        The state had to be saved with [save_state()][magicsoup.world.World.save_state] previously.

        Parameters:
            statedir: Directory that contains all files of that state
            ignore_cell_params: Whether to not update cell parameters as well.
                If you are only interested in the cells' genomes and molecules
                you can set this to `True` to make loading a lot faster.
        """

        self.cell_molecules = torch.load(
            statedir / "cell_molecules.pt", map_location=self.device
        )
        self.cell_map = torch.load(
            statedir / "cell_map.pt", map_location=self.device
        ).bool()
        self.molecule_map = torch.load(
            statedir / "molecule_map.pt", map_location=self.device
        )
        self.cell_survival = torch.load(
            statedir / "cell_survival.pt", map_location=self.device
        ).int()
        self.cell_positions = torch.load(
            statedir / "cell_positions.pt", map_location=self.device
        ).int()
        self.cell_divisions = torch.load(
            statedir / "cell_divisions.pt", map_location=self.device
        ).int()

        with open(statedir / "cells.fasta", "r", encoding="utf-8") as fh:
            text: str = fh.read()
            entries = [d.strip() for d in text.split(">") if len(d.strip()) > 0]

        self.labels = []
        self.genomes = []
        genome_idx_pairs: list[tuple[str, int]] = []
        for idx, entry in enumerate(entries):
            descr, seq = entry.split("\n")
            names = descr.split()
            label = names[1].strip() if len(names) > 1 else ""
            self.genomes.append(seq)
            self.labels.append(label)
            genome_idx_pairs.append((seq, idx))

        self.n_cells = len(genome_idx_pairs)

        if not ignore_cell_params:
            self.kinetics.increase_max_cells(by_n=self.n_cells)
            self.update_cells(genome_idx_pairs=genome_idx_pairs)

    def _divide_cells_as_possible(
        self, parent_idxs: list[int]
    ) -> tuple[list[int], list[int], list[torch.Tensor]]:
        run_idx = self.n_cells
        child_idxs = []
        successful_parent_idxs = []
        new_positions = []
        for parent_idx in parent_idxs:
            pos = self.cell_positions[parent_idx]
            nghbrhd = self._nghbrhd_map[tuple(pos.tolist())]  # type:ignore
            pxls = nghbrhd[~self.cell_map[nghbrhd[:, 0], nghbrhd[:, 1]]]

            n = pxls.size(0)
            if n == 0:
                continue

            new_pos = pxls[random.randint(0, n - 1)]
            new_positions.append(new_pos)

            # immediately set, so that it is already
            # True for the next iteration
            self.cell_map[new_pos[0], new_pos[1]] = True

            self.genomes.append(self.genomes[parent_idx])
            self.labels.append(self.labels[parent_idx])

            successful_parent_idxs.append(parent_idx)
            child_idxs.append(run_idx)
            run_idx += 1

        return successful_parent_idxs, child_idxs, new_positions

    def _find_free_random_positions(self, n_cells: int) -> torch.Tensor:
        # available spots on map
        pxls = torch.argwhere(~self.cell_map)
        n_pxls = pxls.size(0)
        if n_cells > n_pxls:
            n_cells = n_pxls

        # place cells on map
        idxs = random.sample(range(n_pxls), k=n_cells)
        chosen = pxls[idxs]
        return chosen

    def _get_genome_sequences(self, proteome: list[ProteinFact]) -> list[list[str]]:
        proteome = proteome.copy()
        random.shuffle(proteome)

        stop_codons = self.genetics.start_codons
        dom_type_map = self.genetics.domain_types
        one_codon_map = self.genetics.idx_2_one_codon
        two_codon_map = self.genetics.idx_2_two_codon
        react_map = self.kinetics.catal_2_idxs
        trnsp_map = self.kinetics.trnsp_2_idxs
        reg_map = self.kinetics.regul_2_idxs
        sign_map = self.kinetics.sign_2_idxs

        cdss: list[list[str]] = []
        for prot in proteome:
            cds = []
            for dom in prot.domain_facts:
                # Domain structure:
                # 0: domain type definition (1=catalytic, 2=transporter, 3=regulatory)
                # 1-3: 3 x 1-codon specifications (Vmax, Km, sign)
                # 4: 1 x 2-codon specification (reaction, molecule, effector)

                if isinstance(dom, CatalyticDomainFact):
                    react = (tuple(dom.substrates), tuple(dom.products))
                    is_fwd = True
                    if react not in react_map:
                        react = (tuple(dom.products), tuple(dom.substrates))
                        is_fwd = False
                    i3 = random.choice(sign_map[is_fwd])
                    i4 = random.choice(react_map[react])
                    dom_seq = random.choice(dom_type_map[1])
                    sign_seq = one_codon_map[i3]
                    react_seq = two_codon_map[i4]
                    mm_seq = random_genome(s=2 * CODON_SIZE, excl=stop_codons)
                    cds.append(dom_seq + mm_seq + sign_seq + react_seq)

                if isinstance(dom, TransporterDomainFact):
                    i4 = random.choice(trnsp_map[dom.molecule])
                    dom_seq = random.choice(dom_type_map[2])
                    mol_seq = two_codon_map[i4]
                    mm_seq = random_genome(s=2 * CODON_SIZE, excl=stop_codons)
                    sign_seq = random_genome(s=CODON_SIZE, excl=stop_codons)
                    cds.append(dom_seq + mm_seq + sign_seq + mol_seq)

                if isinstance(dom, RegulatoryDomainFact):
                    i4 = random.choice(reg_map[dom.effector])
                    dom_seq = random.choice(dom_type_map[3])
                    mol_seq = two_codon_map[i4]
                    mm_seq = random_genome(s=2 * CODON_SIZE, excl=stop_codons)
                    sign_seq = random_genome(s=CODON_SIZE, excl=stop_codons)
                    cds.append(dom_seq + mm_seq + sign_seq + mol_seq)

            cdss.append(cds)

        return cdss

    def _get_molecule_map(self, n: int, size: int, init: str) -> torch.Tensor:
        args = [n, size, size]
        if init == "zeros":
            return torch.zeros(*args).to(self.device)
        if init == "randn":
            return torch.abs(torch.randn(*args) + 10.0).to(self.device)
        raise ValueError(
            f"Didnt recognize mol_map_init={init}."
            " Should be one of: 'zeros', 'randn'."
        )

    def _get_permeate(self, mol_perm_rate: float) -> float:
        if mol_perm_rate < 0.0:
            mol_perm_rate = -mol_perm_rate

        if mol_perm_rate > 1.0:
            mol_perm_rate = 1.0

        if mol_perm_rate == 0.0:
            return 0.0

        d = 1 / mol_perm_rate
        return 1 / (d + 1)

    def _get_diffuse(self, mol_diff_rate: float) -> torch.nn.Conv2d:
        if mol_diff_rate < 0.0:
            mol_diff_rate = -mol_diff_rate

        # mol_diff_rate > 1.0 could also mean expanding the kernel
        # so that molecules can diffuse more than just 1 pxl per round
        if mol_diff_rate > 1.0:
            mol_diff_rate = 1.0

        if mol_diff_rate == 0.0:
            a = 0.0
            b = 1.0
        else:
            d = 1 / mol_diff_rate
            a = 1 / (d + 8)
            b = d * a
            b = b + 1.0 - (8 * a + b)  # try correcting inaccuracy

        # fmt: off
        kernel = torch.tensor([[[
            [a, a, a],
            [a, b, a],
            [a, a, a],
        ]]]).to(self.device)
        # fmt: on

        conv = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
            device=self.device,
        )
        conv.weight = torch.nn.Parameter(kernel, requires_grad=False)
        return conv

    def _expand_c(self, t: torch.Tensor, n: int) -> torch.Tensor:
        size = t.size()
        zeros = torch.zeros(n, *size[1:], dtype=t.dtype).to(self.device)
        return torch.cat([t, zeros], dim=0)

    def __repr__(self) -> str:
        kwargs = {
            "map_size": self.map_size,
            "abs_temp": self.abs_temp,
            "device": self.device,
            "workers": self.workers,
        }
        args = [f"{k}:{repr(d)}" for k, d in kwargs.items()]
        return f"{type(self).__name__}({','.join(args)})"
