import random
from itertools import product
import math
from io import BytesIO
import pickle
from pathlib import Path
import torch
from magicsoup.constants import CODON_SIZE, DomainSpecType
from magicsoup.util import (
    moore_nghbrhd,
    round_down,
    random_genome,
    randstr,
    closest_value,
)
from magicsoup.containers import (
    Chemistry,
    Molecule,
    Protein,
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
    Likewise, cells are always ordered the same way as in `world.cell_genomes` (see below).
    The index of a certain cell is the index of that cell in `world.cell_genomes`.
    It is the same index as `cell.idx` of a cell object you retrieved with `world.get_cell()`.
    But whenever an operation modifies the number of cells (like `world.kill_cells()` or `world.divide_cells()`),
    cells get new indexes. Here are the most important attributes:

    Attributes:
        cell_genomes: A list of cell genomes. Each cell's index in this list is what is referred to as the cell index.
            The cell index is used for the same cell in other orderings of cells (_e.g._ `labels`, `cell_divisions`, `cell_molecules`).
        labels: List of cell labels. Cells are ordered as in `world.cell_genomes`. Labels are strings that can be used to
            track cell origins. When spawning new cells (`world.spawn_cells()`) a random label is assigned to each cell.
            If a cell divides, its descendants will have the same label.
        cell_map: Boolean 2D tensor referencing which pixels are occupied by a cell.
            Dimension 0 represents the x, dimension 1 y.
        molecule_map: Float 3D tensor describing concentrations (in mM by default) for each molecule species on each pixel in this world.
            Dimension 0 describes the molecule species. They are in the same order as `chemistry.molecules`.
            Dimension 1 represents x, dimension 2 y.
            So, `world.molecule_map[0, 1, 2]` is number of molecules of the 0th molecule species on pixel 1, 2.
        cell_molecules: Float 2D tensor describing concentrations (in mM by default) for each molecule species in each cell.
            Dimension 0 is the cell index. It is the same as in `world.cell_genomes` and the same as on a cell object (`cell.idx`).
            Dimension 1 describes the molecule species. They are in the same order as `chemistry.molecules`.
            So, `world.cell_molecules[0, 1]` represents concentration in mM of the 1st molecule species the 0th cell.
        cell_lifetimes: Integer 1D tensor describing how many time steps each cell survived since the last division.
            This tensor is for monitoring and doesn't have any other effect.
            Cells are in the same as in `world.cell_genomes` and the same as on a cell object (`cell.idx`).
        cell_divisions: Integer 1D tensor describing how many times each cell's ancestors divided.
            This tensor is for monitoring and doesn't have any other effect.
            Cells are in the same order as in `world.cell_genomes` and the same as on a cell object (`cell.idx`).

    Methods for advancing the simulation and to use during a simulation:

    - [spawn_cells()][magicsoup.world.World.spawn_cells] spawn new cells and place them randomly on the map
    - [add_cells()][magicsoup.world.World.add_cells] add previous cells and place them randomly on the map
    - [divide_cells()][magicsoup.world.World.divide_cells] replicate existing cells
    - [update_cells()][magicsoup.world.World.update_cells] update existing cells if their genome has changed
    - [kill_cells()][magicsoup.world.World.kill_cells] kill existing cells
    - [move_cells()][magicsoup.world.World.move_cells] move existing cells to a random position in their Moore's neighborhood
    - [diffuse_molecules()][magicsoup.world.World.diffuse_molecules] let molecules diffuse and permeate by one time step
    - [degrade_molecules()][magicsoup.world.World.degrade_molecules] let molecules degrade by one time step
    - [increment_cell_lifetimes()][magicsoup.world.World.increment_cell_lifetimes] increment `world.cell_lifetimes` by 1
    - [enzymatic_activity()][magicsoup.world.World.enzymatic_activity] let cell proteins work for one time step

    If you want to get a cell with all information about its contents and its current environment use [get_cell()][magicsoup.world.World.get_cell].
    The cell objects returned by this function can be added to a new map using [add_cells()][magicsoup.world.World.add_cells].
    During the simulation you should however work directly with the tensors mentioned above for performance reasons.

    Furthermore, there are methods for saving and loading a simulation.
    For any new simulation use [save()][magicsoup.world.World.save] once to save the whole world object (with chemistry, genetics, kinetics)
    to a pickle file. You can restore it with [from_file()][magicsoup.world.World.from_file] later on.
    Then, to save the world's state you can use [save_state()][magicsoup.world.World.save_state].
    This is a quick, lightweight save, but it only saves things that change during the simulation.
    Use [load_state()][magicsoup.world.World.load_state] to re-load a certain state.

    The `world` object carries `world.genetics`, `world.kinetics`, and `world.chemistry`
    (which is just a reference to the chemistry object that was used when initializing `world`).
    Usually, you don't need to touch them.
    But, if you want to override them or look into some details, see the docstrings of their classes for more information.

    By default in this simulation molecule numbers can be thought of as being in mM, time steps in s, energies J/mol.
    Eventually, they are just numbers and can be interpreted as anything.
    However, with the default parameters in [Kinetics][magicsoup.kinetics.Kinetics]
    and the way [Molecule][magicsoup.containers.Molecule] objects are defined
    it makes sense to interpret them in mM, s, and J.
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
        self.cell_genomes: list[str] = []
        self.cell_labels: list[str] = []
        self.cell_map: torch.Tensor = torch.zeros(map_size, map_size).to(device).bool()
        self.cell_positions: torch.Tensor = torch.zeros(0, 2).to(device).long()
        self.cell_lifetimes: torch.Tensor = torch.zeros(0).to(device).int()
        self.cell_divisions: torch.Tensor = torch.zeros(0).to(device).int()
        self.cell_molecules: torch.Tensor = torch.zeros(0, self.n_molecules).to(device)
        self.molecule_map: torch.Tensor = self._get_molecule_map(
            n=self.n_molecules, size=map_size, init=mol_map_init
        )

    def get_cell(
        self,
        by_idx: int | None = None,
        by_position: tuple[int, int] | None = None,
    ) -> "Cell":
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
        This is not very performant.
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

        pos = self.cell_positions[idx]
        return Cell(
            idx=idx,
            genome=self.cell_genomes[idx],
            position=tuple(pos.tolist()),  # type: ignore
            int_molecules=self.cell_molecules[idx, :],
            ext_molecules=self.molecule_map[:, pos[0], pos[1]],
            label=self.cell_labels[idx],
            n_steps_alive=int(self.cell_lifetimes[idx].item()),
            n_divisions=int(self.cell_divisions[idx].item()),
        )

    def get_neighbors(
        self, cell_idxs: list[int], nghbr_idxs: list[int] | None = None
    ) -> list[tuple[int, int]]:
        """
        For each cell from a list of cell indexes find all other cells that
        are in the Moore's neighborhood of this cell.

        Parameters:
            cell_idxs: Indexes of cells for which to find neighbors
            nghbr_idxs: Optional list of cells regarded as neighbors.
                If `None` (default) each cell in `cell_idxs` can form a pair
                with any other neighboring cell. With `nghbr_idxs` provided
                each cell in `cell_idxs` can only form a pair with a neighboring
                cell that is in `nghbr_idxs`.

        Returns:
            List of tuples of cell indexes for each unique neighboring pair.

        Returned neighbors are unique.
        So, _e.g._ return value `[(1, 4)]` describes the neighbors cell A
        with index 1 and cell B with index 4. `(4, 1)` is not returned.
        """

        if len(cell_idxs) == 0:
            return []

        # rm duplicates to avoid unnecessary compute
        cell_idxs = list(set(cell_idxs))

        if nghbr_idxs is None:
            nghbr_map = self.cell_map
        else:
            nghbr_idxs = list(set(nghbr_idxs))
            nghbr_map = torch.zeros_like(self.cell_map).bool()
            pos = self.cell_positions[nghbr_idxs]
            nghbr_map[pos[:, 0], pos[:, 1]] = True

        nghbrs: list[tuple[int, int]] = []
        for c_idx in cell_idxs:
            c_pos = self.cell_positions[c_idx]
            nghbrhd = self._nghbrhd_map[tuple(c_pos.tolist())]  # type: ignore
            pxls = nghbrhd[nghbr_map[nghbrhd[:, 0], nghbrhd[:, 1]]]
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

        cdss = _get_genome_sequences(
            proteome=proteome,
            domain_types=self.genetics.domain_types,
            stop_codons=self.genetics.stop_codons,
            idx_2_one_codon=self.genetics.idx_2_one_codon,
            idx_2_two_codon=self.genetics.idx_2_two_codon,
            vmax_2_idxs=self.kinetics.vmax_2_idxs,
            km_2_idxs=self.kinetics.km_2_idxs,
            hill_2_idxs=self.kinetics.hill_2_idxs,
            sign_2_idxs=self.kinetics.sign_2_idxs,
            catal_2_idxs=self.kinetics.catal_2_idxs,
            regul_2_idxs=self.kinetics.regul_2_idxs,
            trnsp_2_idxs=self.kinetics.trnsp_2_idxs,
        )
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

    def spawn_cells(self, genomes: list[str], batch_size: int = 1000) -> list[int]:
        """
        Create new cells and place them randomly on the map.
        All lists and tensors that reference cells will be updated.

        Parameters:
            genomes: List of genomes of the newly added cells
            batch_size: Batch size used for updating cell kinetics. Reduce this number
                to reduce memory required during update.

        Returns:
            The indexes of successfully added cells.

        Each cell will be placed randomly on the map and receive half the molecules of the pixel where it was added.
        If there are less pixels left on the cell map than cells you want to add,
        only the remaining pixels will be filled with new cells.
        Each cell will also receive a random label.
        """
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

        new_pos = free_pos[:n_new_cells]
        new_idxs = list(range(self.n_cells, self.n_cells + n_new_cells))
        self.n_cells += n_new_cells
        self.cell_genomes.extend(genomes)
        self.cell_labels.extend(randstr(n=12) for _ in range(n_new_cells))

        self.cell_lifetimes = self._expand_c(t=self.cell_lifetimes, n=n_new_cells)
        self.cell_positions = self._expand_c(t=self.cell_positions, n=n_new_cells)
        self.cell_divisions = self._expand_c(t=self.cell_divisions, n=n_new_cells)
        self.cell_molecules = self._expand_c(t=self.cell_molecules, n=n_new_cells)
        self.kinetics.increase_max_cells(by_n=n_new_cells)

        # occupy positions
        xs = new_pos[:, 0]
        ys = new_pos[:, 1]
        self.cell_map[xs, ys] = True
        self.cell_positions[new_idxs] = new_pos

        # cell is picking up half the molecules of the pxl it is born on
        pickup = self.molecule_map[:, xs, ys] * 0.5
        self.cell_molecules[new_idxs, :] += pickup.T
        self.molecule_map[:, xs, ys] -= pickup

        proteomes = self.genetics.translate_genomes(genomes=genomes)

        set_proteomes = []
        set_idxs = []
        for proteome, idx in zip(proteomes, new_idxs):
            if len(proteome) > 0:
                set_proteomes.append([d[0] for d in proteome])
                set_idxs.append(idx)

        n_new_proteomes = len(set_proteomes)
        if n_new_proteomes == 0:
            return []

        n_max_prots = max(len(d) for d in set_proteomes)
        self.kinetics.increase_max_proteins(max_n=n_max_prots)

        for a in range(0, n_new_proteomes, batch_size):
            b = a + batch_size
            self.kinetics.set_cell_params(
                cell_idxs=set_idxs[a:b], proteomes=set_proteomes[a:b]
            )

        return new_idxs

    def add_cells(self, cells: list["Cell"], batch_size: int = 1000) -> list[int]:
        """
        Place cells randomly on the map.
        All lists and tensors that reference cells will be updated.

        Parameters:
            cells: List of cells to be added
            batch_size: Batch size used for updating cell kinetics. Reduce this number
                to reduce memory required during update.

        Returns:
            The indexes of successfully added cells.

        Each cell will be placed randomly on the map.
        If there are less pixels left on the cell map than cells you want to add,
        only the remaining pixels will be filled with cells.
        They keep their genomes, proteomes, intracellular molecule compositions, lifetimes, and divisions.
        Cell indexes and positions will be new.
        """
        n_new_cells = len(cells)
        if n_new_cells == 0:
            return []

        free_pos = self._find_free_random_positions(n_cells=n_new_cells)
        n_avail_pos = free_pos.size(0)
        if n_avail_pos == 0:
            return []

        if n_avail_pos < n_new_cells:
            n_new_cells = n_avail_pos
            random.shuffle(cells)
            cells = cells[:n_new_cells]

        new_pos = free_pos[:n_new_cells]
        new_idxs = list(range(self.n_cells, self.n_cells + n_new_cells))
        self.n_cells += n_new_cells
        for cell in cells:
            self.cell_genomes.append(cell.genome)
            self.cell_labels.append(cell.label)

        self.cell_lifetimes = self._expand_c(t=self.cell_lifetimes, n=n_new_cells)
        self.cell_positions = self._expand_c(t=self.cell_positions, n=n_new_cells)
        self.cell_divisions = self._expand_c(t=self.cell_divisions, n=n_new_cells)
        self.cell_molecules = self._expand_c(t=self.cell_molecules, n=n_new_cells)
        self.kinetics.increase_max_cells(by_n=n_new_cells)

        # occupy positions
        xs = new_pos[:, 0]
        ys = new_pos[:, 1]
        self.cell_map[xs, ys] = True
        self.cell_positions[new_idxs] = new_pos

        # previous molecules, lifetimes, divisions are transfered
        int_mols = []
        lifetimes = []
        divisions = []
        genomes = []
        for cell in cells:
            int_mols.append(cell.int_molecules)
            lifetimes.append(cell.n_steps_alive)
            divisions.append(cell.n_divisions)
            genomes.append(cell.genome)

        self.cell_molecules[new_idxs, :] = torch.stack(int_mols).to(self.device).float()
        self.cell_lifetimes[new_idxs] = torch.tensor(lifetimes).to(self.device).int()
        self.cell_divisions[new_idxs] = torch.tensor(divisions).to(self.device).int()

        proteomes = self.genetics.translate_genomes(genomes=genomes)

        set_proteomes = []
        set_idxs = []
        for proteome, idx in zip(proteomes, new_idxs):
            if len(proteome) > 0:
                set_proteomes.append([d[0] for d in proteome])
                set_idxs.append(idx)

        n_new_proteomes = len(set_proteomes)
        if n_new_proteomes == 0:
            return []

        n_max_prots = max(len(d) for d in set_proteomes)
        self.kinetics.increase_max_proteins(max_n=n_max_prots)

        for a in range(0, n_new_proteomes, batch_size):
            b = a + batch_size
            self.kinetics.set_cell_params(
                cell_idxs=set_idxs[a:b], proteomes=set_proteomes[a:b]
            )

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

        self.cell_lifetimes = self._expand_c(t=self.cell_lifetimes, n=n_new_cells)
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
        self.cell_lifetimes[descendant_idxs] = 0

        return list(zip(parent_idxs, child_idxs))

    def update_cells(
        self, genome_idx_pairs: list[tuple[str, int]], batch_size: int = 1000
    ):
        """
        Update existing cells with new genomes.

        Parameters:
            genome_idx_pairs: List of tuples of genomes and cell indexes
            batch_size: Batch size used for updating cell kinetics. Reduce this number
                to reduce memory required during update.

        The indexes refer to the index of each cell that is changed.
        The genomes refer to the genome of each cell that is changed.
        """
        if len(genome_idx_pairs) == 0:
            return

        genomes = [d[0] for d in genome_idx_pairs]
        proteomes = self.genetics.translate_genomes(genomes=genomes)

        set_idxs: list[int] = []
        unset_idxs: list[int] = []
        set_proteomes: list[list[list[DomainSpecType]]] = []
        for (genome, idx), proteome in zip(genome_idx_pairs, proteomes):
            self.cell_genomes[idx] = genome
            if len(proteome) > 0:
                set_idxs.append(idx)
                set_proteomes.append([d[0] for d in proteome])
            else:
                unset_idxs.append(idx)

        self.kinetics.unset_cell_params(cell_idxs=unset_idxs)

        n_set_proteomes = len(set_proteomes)
        if n_set_proteomes > 0:
            max_prots = max(len(d) for d in set_proteomes)
            self.kinetics.increase_max_proteins(max_n=max_prots)
            for a in range(0, n_set_proteomes, batch_size):
                b = a + batch_size
                self.kinetics.set_cell_params(
                    cell_idxs=set_idxs[a:b], proteomes=set_proteomes[a:b]
                )

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
        if len(cell_idxs) == 0:
            return

        # duplicates could raise error later
        cell_idxs = list(set(cell_idxs))

        xs = self.cell_positions[cell_idxs, 0]
        ys = self.cell_positions[cell_idxs, 1]
        self.cell_map[xs, ys] = False

        spillout = self.cell_molecules[cell_idxs, :]
        self.molecule_map[:, xs, ys] += spillout.T

        n_cells = self.cell_lifetimes.size(0)
        keep = torch.ones(n_cells, dtype=torch.bool).to(self.device)
        keep[cell_idxs] = False
        self.cell_lifetimes = self.cell_lifetimes[keep]
        self.cell_positions = self.cell_positions[keep]
        self.cell_divisions = self.cell_divisions[keep]
        self.cell_molecules = self.cell_molecules[keep]
        self.kinetics.remove_cell_params(keep=keep)

        for idx in sorted(cell_idxs, reverse=True):
            self.cell_genomes.pop(idx)
            self.cell_labels.pop(idx)

        self.n_cells -= len(cell_idxs)

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
        if len(cell_idxs) == 0:
            return

        # duplicates could lead to unexpected results
        cell_idxs = list(set(cell_idxs))

        old_xs = self.cell_positions[cell_idxs, 0]
        old_ys = self.cell_positions[cell_idxs, 1]
        self.cell_map[old_xs, old_ys] = False

        new_pos = self._find_free_random_positions(n_cells=len(cell_idxs))
        new_xs = new_pos[:, 0]
        new_ys = new_pos[:, 1]

        self.cell_map[new_xs, new_ys] = True
        self.cell_positions[cell_idxs] = new_pos

    def enzymatic_activity(self):
        """
        Catalyze reactions for one time step (1s by default).
        This includes molecule transport into or out of the cell.
        `world.molecule_map` and `world.cell_molecules` are updated.
        """
        if self.n_cells == 0:
            return

        xs = self.cell_positions[:, 0]
        ys = self.cell_positions[:, 1]
        X0 = torch.cat([self.cell_molecules, self.molecule_map[:, xs, ys].T], dim=1)
        X1 = self.kinetics.integrate_signals(X=X0)

        self.molecule_map[:, xs, ys] = X1[:, self._ext_mol_idxs].T
        self.cell_molecules = X1[:, self._int_mol_idxs]

    @torch.no_grad()
    def diffuse_molecules(self):
        """
        Let molecules in molecule map diffuse and permeate through cell membranes
        by one time step (1s by default).
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

    def increment_cell_lifetimes(self):
        """
        Increment `world.cell_lifetimes` by 1.
        This is for monitoring and doesn't have any other effect.
        """
        self.cell_lifetimes += 1

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
        torch.save(self.cell_lifetimes, statedir / "cell_lifetimes.pt")
        torch.save(self.cell_positions, statedir / "cell_positions.pt")
        torch.save(self.cell_divisions, statedir / "cell_divisions.pt")

        lines: list[str] = []
        for idx, (genome, label) in enumerate(zip(self.cell_genomes, self.cell_labels)):
            lines.append(f">{idx} {label}\n{genome}")

        with open(statedir / "cells.fasta", "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    def load_state(
        self, statedir: Path, ignore_cell_params: bool = False, batch_size: int = 1000
    ):
        """
        Load a saved world state.
        The state had to be saved with [save_state()][magicsoup.world.World.save_state] previously.

        Parameters:
            statedir: Directory that contains all files of that state
            ignore_cell_params: Whether to not update cell parameters as well.
                If you are only interested in the cells' genomes and molecules
                you can set this to `True` to make loading a lot faster.
            batch_size: Batch size used for updating cell kinetics. Reduce this number
                to reduce memory required during update. This is irrelevant with
                `ignore_cell_params=True`.
        """
        self.kill_cells(cell_idxs=list(range(self.n_cells)))

        self.cell_molecules = torch.load(
            statedir / "cell_molecules.pt", map_location=self.device
        )
        self.cell_map = torch.load(
            statedir / "cell_map.pt", map_location=self.device
        ).bool()
        self.molecule_map = torch.load(
            statedir / "molecule_map.pt", map_location=self.device
        )
        self.cell_lifetimes = torch.load(
            statedir / "cell_lifetimes.pt", map_location=self.device
        ).int()
        self.cell_positions = torch.load(
            statedir / "cell_positions.pt", map_location=self.device
        ).long()
        self.cell_divisions = torch.load(
            statedir / "cell_divisions.pt", map_location=self.device
        ).int()

        with open(statedir / "cells.fasta", "r", encoding="utf-8") as fh:
            text: str = fh.read()
            entries = [d.strip() for d in text.split(">") if len(d.strip()) > 0]

        self.cell_labels = []
        self.cell_genomes = []
        genome_idx_pairs: list[tuple[str, int]] = []
        for idx, entry in enumerate(entries):
            parts = entry.split("\n")
            descr = parts[0]
            seq = "" if len(parts) < 2 else parts[1]
            names = descr.split()
            label = names[1].strip() if len(names) > 1 else ""
            self.cell_genomes.append(seq)
            self.cell_labels.append(label)
            genome_idx_pairs.append((seq, idx))

        self.n_cells = len(genome_idx_pairs)

        if not ignore_cell_params:
            self.kinetics.increase_max_cells(by_n=self.n_cells)
            self.update_cells(genome_idx_pairs=genome_idx_pairs, batch_size=batch_size)

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

            self.cell_genomes.append(self.cell_genomes[parent_idx])
            self.cell_labels.append(self.cell_labels[parent_idx])

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


class Cell:
    """
    Object representing a cell with its environment.

    Arguments:
        genome: Full genome sequence of this cell.
        int_molecules: Intracellular molecules. A tensor with one dimension that represents
            each molecule species in the same order as defined in [Chemistry][magicsoup.containers.Chemistry].
        ext_molecules: Extracellular molecules. A tensor with one dimension that represents
            each molecule species in the same order as defined in [Chemistry][magicsoup.containers.Chemistry].
            These are the molecules of the pixel the cell is currently living on.
        position: Position on the cell map.
        idx: The current index of this cell.
        label: Label which can be used to track cells. Has no effect.
        n_steps_alive: Number of time steps this cell has lived since last division.
        n_divisions: Number of times this cell's ancestors already divided.

    Usually, you wouldn't instantiate this object.
    You get it when calling [get_cell()][magicsoup.world.World.get_cell] after you have spawned some
    cells to a world (via [spawn_cells()][magicsoup.world.World.spawn_cells]).
    On the `world` object all cells are actually represented as a combination of lists and tensors.
    [get_cell()][magicsoup.world.World.get_cell] gathers all information for one cell and
    represents it in this `Cell` object.

    When a cell replicates its genome and proteome are copied.
    Both descendants will recieve half of all molecules each.
    Both their `n_divisions` attributes are incremented.
    The cell's `label` will be copied as well.
    This way you can track cells' origins.
    """

    def __init__(
        self,
        genome: str,
        int_molecules: torch.Tensor,
        ext_molecules: torch.Tensor,
        position: tuple[int, int] = (-1, -1),
        idx: int = -1,
        label: str = "C",
        n_steps_alive: int = 0,
        n_divisions: int = 0,
    ):
        self.genome = genome
        self.label = label
        self.int_molecules = int_molecules
        self.ext_molecules = ext_molecules
        self.position = position
        self.idx = idx
        self.n_steps_alive = n_steps_alive
        self.n_divisions = n_divisions

    def get_proteome(self, world: World) -> list[Protein]:
        """
        Get a representation of the cell's proteome as a list of Protein objects
        """
        (cdss,) = world.genetics.translate_genomes(genomes=[self.genome])
        return world.kinetics.get_proteome(proteome=cdss) if len(cdss) > 0 else []

    def copy(self, **kwargs) -> "Cell":
        old_kwargs = {
            "genome": self.genome,
            "position": self.position,
            "idx": self.idx,
            "label": self.label,
            "n_steps_alive": self.n_steps_alive,
            "n_divisions": self.n_divisions,
        }
        return Cell(**{**old_kwargs, **kwargs})  # type: ignore

    def __repr__(self) -> str:
        kwargs = {
            "genome": self.genome,
            "position": self.position,
            "idx": self.idx,
            "label": self.label,
            "n_steps_alive": self.n_steps_alive,
            "n_divisions": self.n_divisions,
        }
        args = [f"{k}:{repr(d)}" for k, d in kwargs.items()]
        return f"{type(self).__name__}({','.join(args)})"


def _get_genome_sequences(
    proteome: list[ProteinFact],
    domain_types: dict[int, list[str]],
    stop_codons: list[str],
    idx_2_one_codon: dict[int, str],
    idx_2_two_codon: dict[int, str],
    km_2_idxs: dict[float, list[int]],
    hill_2_idxs: dict[int, list[int]],
    vmax_2_idxs: dict[float, list[int]],
    sign_2_idxs: dict[bool, list[int]],
    catal_2_idxs: dict[tuple[tuple[Molecule, ...], tuple[Molecule, ...]], list[int]],
    trnsp_2_idxs: dict[Molecule, list[int]],
    regul_2_idxs: dict[tuple[Molecule, bool], list[int]],
) -> list[list[str]]:
    # spec: dom_type, idx0, idx1, idx2, idx3
    # dom_type = 2 codon, idx0-2 = 1 codon, idx3 = 2 codons
    # domain_types: type int 1-3 -> list of 1 codon strs (no stop)
    # idx_2_one_codon: index int -> 1 codon str (no stop)
    # idx_2_two_codon: index int -> 2 codons str (2nd can be stop)
    # *_2_idxs: mapping of meaningful values to idxs
    proteome = proteome.copy()
    random.shuffle(proteome)

    cdss: list[list[str]] = []
    for prot in proteome:
        cds = []
        for dom in prot.domain_facts:
            # Domain structure:
            # 0: domain type definition (1=catalytic, 2=transporter, 3=regulatory)
            # 1-3: 3 x 1-codon specifications (Vmax, Km, sign, hill)
            # 4: 1 x 2-codon specification (reaction, molecule, effector)

            if isinstance(dom, CatalyticDomainFact):
                seq = _get_catalytic_domain_sequence(
                    dom=dom,
                    domain_types=domain_types,
                    stop_codons=stop_codons,
                    sign_2_idxs=sign_2_idxs,
                    km_2_idxs=km_2_idxs,
                    vmax_2_idxs=vmax_2_idxs,
                    catal_2_idxs=catal_2_idxs,
                    idx_2_one_codon=idx_2_one_codon,
                    idx_2_two_codon=idx_2_two_codon,
                )
                cds.append(seq)

            if isinstance(dom, TransporterDomainFact):
                seq = _get_transporter_domain_sequence(
                    dom=dom,
                    domain_types=domain_types,
                    stop_codons=stop_codons,
                    sign_2_idxs=sign_2_idxs,
                    km_2_idxs=km_2_idxs,
                    vmax_2_idxs=vmax_2_idxs,
                    trnsp_2_idxs=trnsp_2_idxs,
                    idx_2_one_codon=idx_2_one_codon,
                    idx_2_two_codon=idx_2_two_codon,
                )
                cds.append(seq)

            if isinstance(dom, RegulatoryDomainFact):
                seq = _get_regulatory_domain_sequence(
                    dom=dom,
                    domain_types=domain_types,
                    stop_codons=stop_codons,
                    sign_2_idxs=sign_2_idxs,
                    km_2_idxs=km_2_idxs,
                    hill_2_idxs=hill_2_idxs,
                    regul_2_idxs=regul_2_idxs,
                    idx_2_one_codon=idx_2_one_codon,
                    idx_2_two_codon=idx_2_two_codon,
                )
                cds.append(seq)

        cdss.append(cds)

    return cdss


def _get_regulatory_domain_sequence(
    dom: RegulatoryDomainFact,
    domain_types: dict[int, list[str]],
    stop_codons: list[str],
    sign_2_idxs: dict[bool, list[int]],
    km_2_idxs: dict[float, list[int]],
    hill_2_idxs: dict[int, list[int]],
    regul_2_idxs: dict[tuple[Molecule, bool], list[int]],
    idx_2_one_codon: dict[int, str],
    idx_2_two_codon: dict[int, str],
) -> str:
    # regulatory domain type: 3
    # idx0: hill coefficient (1 codon, no stop)
    # idx1: Km (1 codon, no stop)
    # idx2: sign (1 codon, no stop)
    # idx3: effector (2 codon, 2nd can be stop)
    # is_transmembrane defined by effector (int/ext molecules)
    dom_seq = random.choice(domain_types[3])

    if dom.hill is not None:
        val = closest_value(values=hill_2_idxs, key=dom.hill)
        i0 = random.choice(hill_2_idxs[int(val)])
        i0_seq = idx_2_one_codon[i0]
    else:
        i0_seq = random_genome(s=CODON_SIZE, excl=stop_codons)

    if dom.km is not None:
        val = closest_value(values=km_2_idxs, key=dom.km)
        i1 = random.choice(km_2_idxs[val])
        i1_seq = idx_2_one_codon[i1]
    else:
        i1_seq = random_genome(s=CODON_SIZE, excl=stop_codons)

    if dom.is_inhibiting is not None:
        i2 = random.choice(sign_2_idxs[not dom.is_inhibiting])
        i2_seq = idx_2_one_codon[i2]
    else:
        i2_seq = random_genome(s=CODON_SIZE, excl=stop_codons)

    i3 = random.choice(regul_2_idxs[(dom.effector, dom.is_transmembrane)])
    i3_seq = idx_2_two_codon[i3]

    return dom_seq + i0_seq + i1_seq + i2_seq + i3_seq


def _get_transporter_domain_sequence(
    dom: TransporterDomainFact,
    domain_types: dict[int, list[str]],
    stop_codons: list[str],
    sign_2_idxs: dict[bool, list[int]],
    km_2_idxs: dict[float, list[int]],
    vmax_2_idxs: dict[float, list[int]],
    trnsp_2_idxs: dict[Molecule, list[int]],
    idx_2_one_codon: dict[int, str],
    idx_2_two_codon: dict[int, str],
) -> str:
    # transporter domain type: 2
    # idx0: Vmax (1 codon, no stop)
    # idx1: Km (1 codon, no stop)
    # idx2: is_exporter (1 codon, no stop)
    # idx3: molecule (2 codon, 2nd can be stop)
    dom_seq = random.choice(domain_types[2])

    if dom.vmax is not None:
        val = closest_value(values=vmax_2_idxs, key=dom.vmax)
        i0 = random.choice(vmax_2_idxs[val])
        i0_seq = idx_2_one_codon[i0]
    else:
        i0_seq = random_genome(s=CODON_SIZE, excl=stop_codons)

    if dom.km is not None:
        val = closest_value(values=km_2_idxs, key=dom.km)
        i1 = random.choice(km_2_idxs[val])
        i1_seq = idx_2_one_codon[i1]
    else:
        i1_seq = random_genome(s=CODON_SIZE, excl=stop_codons)

    if dom.is_exporter is not None:
        i2 = random.choice(sign_2_idxs[dom.is_exporter])
        i2_seq = idx_2_one_codon[i2]
    else:
        i2_seq = random_genome(s=CODON_SIZE, excl=stop_codons)

    i3 = random.choice(trnsp_2_idxs[dom.molecule])
    i3_seq = idx_2_two_codon[i3]

    return dom_seq + i0_seq + i1_seq + i2_seq + i3_seq


def _get_catalytic_domain_sequence(
    dom: CatalyticDomainFact,
    domain_types: dict[int, list[str]],
    stop_codons: list[str],
    sign_2_idxs: dict[bool, list[int]],
    km_2_idxs: dict[float, list[int]],
    vmax_2_idxs: dict[float, list[int]],
    catal_2_idxs: dict[tuple[tuple[Molecule, ...], tuple[Molecule, ...]], list[int]],
    idx_2_one_codon: dict[int, str],
    idx_2_two_codon: dict[int, str],
) -> str:
    # catalytic domain type: 1
    # idx0: Vmax (1 codon, no stop)
    # idx1: Km (1 codon, no stop)
    # idx2: direction (1 codon, no stop)
    # idx3: reaction (2 codon, 2nd can be stop)
    dom_seq = random.choice(domain_types[1])

    if dom.vmax is not None:
        val = closest_value(values=vmax_2_idxs, key=dom.vmax)
        i0 = random.choice(vmax_2_idxs[val])
        i0_seq = idx_2_one_codon[i0]
    else:
        i0_seq = random_genome(s=CODON_SIZE, excl=stop_codons)

    if dom.km is not None:
        val = closest_value(values=km_2_idxs, key=dom.km)
        i1 = random.choice(km_2_idxs[val])
        i1_seq = idx_2_one_codon[i1]
    else:
        i1_seq = random_genome(s=CODON_SIZE, excl=stop_codons)

    react = (tuple(dom.substrates), tuple(dom.products))
    is_fwd = True
    if react not in catal_2_idxs:
        react = (tuple(dom.products), tuple(dom.substrates))
        is_fwd = False
    i2 = random.choice(sign_2_idxs[is_fwd])
    i2_seq = idx_2_one_codon[i2]
    i3 = random.choice(catal_2_idxs[react])
    i3_seq = idx_2_two_codon[i3]

    return dom_seq + i0_seq + i1_seq + i2_seq + i3_seq
