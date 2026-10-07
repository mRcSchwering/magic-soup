import logging
import pickle
import random
from io import BytesIO
from pathlib import Path

import torch

from magicsoup.biology import Cell
from magicsoup.cells import Cells, CellsKwargs
from magicsoup.chemistry import Chemistry
from magicsoup.culture import Culture, CultureKwargs
from magicsoup.genomics import Genomics, GenomicsKwargs
from magicsoup.kinetics import Kinetics, KineticsKwargs
from magicsoup.mutations import point_mutations, recombinations
from magicsoup.proteomics import Proteomics, ProteomicsKwargs
from magicsoup.util import TensorClass, randstr

_log = logging.getLogger(__name__)


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


# TODO: see if tensors can be used wherever indices are passed around
#       only strs need to be lists


class World(TensorClass):

    def __init__(
        self,
        chemistry: Chemistry,
        kinetics: Kinetics | None = None,
        culture: Culture | None = None,
        cells: Cells | None = None,
        time_step: float = 1.0,
        kinetics_kwargs: KineticsKwargs | None = None,
        culture_kwargs: CultureKwargs | None = None,
        cells_kwargs: CellsKwargs | None = None,
        genomics_kwargs: GenomicsKwargs | None = None,
        proteomics_kwargs: ProteomicsKwargs | None = None,
        device: str = "cpu",
        ftype: torch.dtype = torch.float32,
        itype: torch.dtype = torch.int8,
    ):
        super().__init__(device=device, itype=itype, ftype=ftype)

        self.time_step = time_step
        self.chemistry = chemistry
        self.kinetics = kinetics or Kinetics(
            **(kinetics_kwargs or {}),
            device=device,
            itype=itype,
            ftype=ftype,
        )
        self.culture = culture or (
            Culture(
                **(culture_kwargs or {}),
                chemistry=chemistry,
                device=device,
                itype=itype,
                ftype=ftype,
            )
        )

        if not cells:
            genomics = Genomics(**(genomics_kwargs or {}))
            proteomics = Proteomics(
                **(proteomics_kwargs or {}),
                chemistry=chemistry,
                scalar_enc_size=max(genomics.one_codon_map.values()),
                vector_enc_size=max(genomics.two_codon_map.values()),
                device=device,
                itype=itype,
                ftype=ftype,
            )
            self.cells = Cells(
                **(cells_kwargs or {}),
                c_max=self.culture.pixels,
                chemistry=chemistry,
                genomics=genomics,
                proteomics=proteomics,
                device=device,
                itype=itype,
                ftype=ftype,
            )
        else:
            self.cells = cells

        n_molecules = len(chemistry.molecules)
        self._int_mol_idxs = list(range(n_molecules))
        self._ext_mol_idxs = list(range(n_molecules, n_molecules * 2))

    def get_cell(
        self,
        by_idx: int | None = None,
        by_position: tuple[int, int] | None = None,
    ) -> "Cell":
        idx = -1
        if by_idx is not None:
            idx = by_idx
        if by_position is not None:
            pos = self.idxtensor(by_position)
            mask = (self.cells.positions == pos).all(dim=1)
            idxs = torch.argwhere(mask).flatten().tolist()
            if len(idxs) == 0:
                raise ValueError(f"Cell at {by_position} not found")
            idx = idxs[0]

        return Cell(
            world=self,
            idx=idx,
            genome=self.cells.genomes.items[idx],
            position=tuple(self.cells.positions[idx].tolist()),  # type: ignore
            label=self.cells.labels.items[idx],
            age=int(self.cells.ages[idx].item()),
            generation=int(self.cells.generations[idx].item()),
        )

    def get_neighbors(
        self, cell_idxs: list[int], nghbr_idxs: list[int] | None = None
    ) -> list[tuple[int, int]]:
        if len(cell_idxs) == 0:
            return []

        from_idxs = list(set(cell_idxs))
        if nghbr_idxs is None:
            to_idxs = list(set(cell_idxs))
        else:
            to_idxs = list(set(nghbr_idxs))

        if to_idxs == 0:
            return []

        xs = self.cells.positions[:, 0].tolist()
        ys = self.cells.positions[:, 1].tolist()
        positions = [(x, y) for x, y in zip(xs, ys)]
        nghbrs = self.culture.get_neighbors(
            from_idxs=from_idxs, to_idxs=to_idxs, positions=positions
        )
        return nghbrs

    def spawn_cells(self, genomes: list[str]) -> torch.Tensor:
        n_new_cells = len(genomes)
        if n_new_cells == 0:
            return self.idxtensor([])

        free_pos = self.culture.find_free_random_positions(n=n_new_cells)
        n_avail_pos = free_pos.size(0)
        if n_avail_pos == 0:
            _log.warning("No free positions to spawn %d cells", n_new_cells)
            return self.idxtensor([])

        if n_avail_pos < n_new_cells:
            _log.warning(
                "Only %d free positions to spawn %d cells", n_avail_pos, n_new_cells
            )
            n_new_cells = n_avail_pos
            random.shuffle(genomes)
            genomes = genomes[:n_new_cells]

        n_living_cells = self.cells.get_alive_cells()
        self.cells.update_c(c_req=n_living_cells + n_new_cells)
        free_idxs = self.cells.get_available_idxs()
        new_idxs = free_idxs[:n_new_cells]

        # occupy positions
        new_pos = free_pos[:n_new_cells]
        xs = new_pos[:, 0]
        ys = new_pos[:, 1]
        self.culture.cell_map[xs, ys] = True
        self.cells.positions[new_idxs] = new_pos

        # cell is picking up half the molecules of the pxl it is born on
        pickup = self.culture.molecule_map[:, xs, ys] * 0.5
        self.cells.molecules[new_idxs, :] = pickup.T
        self.culture.molecule_map[:, xs, ys] = pickup

        # update parameters from genomes
        self.cells.update_genomes(genomes=genomes, idxs=new_idxs)

        # update other values
        self.cells.alive[new_idxs] = True
        self.cells.labels[new_idxs] = [randstr(n=12) for _ in range(n_new_cells)]

        return new_idxs

    def add_cells(self, cells: list["Cell"]) -> torch.Tensor:
        n_new_cells = len(cells)
        if n_new_cells == 0:
            return self.idxtensor([])

        free_pos = self.culture.find_free_random_positions(n=n_new_cells)
        n_avail_pos = free_pos.size(0)
        if n_avail_pos == 0:
            _log.warning("No free positions to add %d cells", n_new_cells)
            return self.idxtensor([])

        if n_avail_pos < n_new_cells:
            _log.warning(
                "Only %d free positions to add %d cells", n_avail_pos, n_new_cells
            )
            n_new_cells = n_avail_pos
            random.shuffle(cells)
            cells = cells[:n_new_cells]

        n_cells = self.cells.get_alive_cells()
        self.cells.update_c(c_req=n_cells + n_new_cells)
        free_idxs = self.cells.get_available_idxs()
        new_idxs = free_idxs[:n_new_cells]

        # occupy positions
        new_pos = free_pos[:n_new_cells]
        xs = new_pos[:, 0]
        ys = new_pos[:, 1]
        self.culture.cell_map[xs, ys] = True
        self.cells.positions[new_idxs] = new_pos

        # update parameters from genomes
        self.cells.update_genomes(genomes=[d.genome for d in cells], idxs=new_idxs)

        # update other parameters
        int_mols = [d.int_molecules for d in cells]
        self.cells.alive[new_idxs] = True
        self.cells.labels[new_idxs] = [d.label for d in cells]
        self.cells.molecules[new_idxs, :] = self.ftensor(torch.stack(int_mols))
        self.cells.ages[new_idxs] = self.ftensor([d.age for d in cells])
        self.cells.generations[new_idxs] = self.itensor([d.generation for d in cells])

        return new_idxs

    def divide_cells(self, cell_idxs: torch.Tensor) -> list[tuple[int, int]]:
        # already places children on cell map
        parent_idxs, child_positions = self.culture.divide_cells(
            cell_idxs=cell_idxs, cell_positions=self.cells.positions
        )

        n_cells = self.cells.get_alive_cells()
        n_new_cells = parent_idxs.numel()

        if n_new_cells == 0:
            return []

        self.cells.update_c(c_req=n_cells)
        child_idxs = self.cells.get_available_idxs()[:n_new_cells]

        # update cell parameters for children
        self.cells.alive[child_idxs] = True
        self.cells.labels[child_idxs] = self.cells.labels[parent_idxs]
        self.cells.positions[child_idxs] = child_positions

        # update child kinetics parameters from genomes
        genomes = self.cells.genomes[parent_idxs]
        self.cells.update_genomes(genomes=genomes, idxs=child_idxs)

        # cells share molecules, increment generations, reset lifetimes
        descendant_idxs = torch.cat([parent_idxs, child_idxs])
        self.cells.molecules[child_idxs] = self.cells.molecules[parent_idxs]
        self.cells.molecules[descendant_idxs] *= 0.5
        self.cells.generations[child_idxs] = self.cells.generations[parent_idxs]
        self.cells.generations[descendant_idxs] += 1
        self.cells.ages[descendant_idxs] = 0.0

        return list(zip(parent_idxs, child_idxs))

    def update_cells(self, genome_idx_pairs: list[tuple[str, int]]) -> None:
        if len(genome_idx_pairs) == 0:
            return

        genomes = [d[0] for d in genome_idx_pairs]
        idxs = self.idxtensor([d[1] for d in genome_idx_pairs])
        self.cells.update_genomes(genomes=genomes, idxs=idxs)

    def kill_cells(self, cell_idxs: list[int] | None = None) -> None:
        if cell_idxs is None:
            cell_idxs = self.cells.get_cell_idxs().tolist()

        if len(cell_idxs) == 0:
            return

        # duplicates could raise error later
        cell_idxs = list(set(cell_idxs))

        # free up map
        xs = self.cells.positions[cell_idxs, 0]
        ys = self.cells.positions[cell_idxs, 1]
        self.culture.cell_map[xs, ys] = False

        # spill out molecules
        spillout = self.cells.molecules[cell_idxs, :]
        self.culture.molecule_map[:, xs, ys] += spillout.T

        # unset parameters
        self.cells.genomes[cell_idxs] = [""] * len(cell_idxs)
        self.cells.labels[cell_idxs] = [""] * len(cell_idxs)
        self.cells.ages[cell_idxs] = 0.0
        self.cells.positions[cell_idxs] = -1
        self.cells.generations[cell_idxs] = 0
        self.cells.molecules[cell_idxs] = 0.0

        self.cells.v_max[cell_idxs] = 0.0
        self.cells.k_f[cell_idxs] = 0.0
        self.cells.k_b[cell_idxs] = 0.0
        self.cells.K_r[cell_idxs] = 0.0
        self.cells.N_f[cell_idxs] = 0
        self.cells.N_b[cell_idxs] = 0
        self.cells.N_h[cell_idxs] = 0
        self.cells.N[cell_idxs] = 0

    def migrate_cells(self, cell_idxs: torch.Tensor | None = None) -> None:
        if cell_idxs is None:
            cell_idxs = self.cells.get_cell_idxs()

        if len(cell_idxs) == 0:
            return

        # positions are updated in place
        self.culture.migrate_cells(
            cell_idxs=cell_idxs, cell_positions=self.cells.positions
        )

    def resuspend_cells(self, cell_idxs: list[int] | None = None) -> None:
        n_cells = self.cells.get_alive_cells()

        # unoccupy current positions
        old_xs = self.cells.positions[:, 0]
        old_ys = self.cells.positions[:, 1]
        self.culture.cell_map[old_xs, old_ys] = False

        # find new unoccupied positions
        new_pos = self.culture.find_free_random_positions(n=n_cells)
        new_xs = new_pos[:, 0]
        new_ys = new_pos[:, 1]

        self.culture.cell_map[new_xs, new_ys] = True
        self.cells.positions[cell_idxs] = new_pos

    def enzymatic_activity(self) -> None:
        alive = self.cells.alive
        xs = self.cells.positions[alive, 0]
        ys = self.cells.positions[alive, 1]

        # collect internal and external molecules for x0
        x0 = torch.cat(
            [self.cells.molecules[alive], self.culture.molecule_map[:, xs, ys].T],
            dim=1,
        )

        # integrate
        x1 = self.kinetics.step_protein_activity(
            h=self.time_step,
            x0=x0,
            N=self.cells.N[alive],
            N_f=self.cells.N_f[alive],
            N_b=self.cells.N_b[alive],
            N_h=self.cells.N_h[alive],
            k_f=self.cells.k_f[alive],
            k_b=self.cells.k_b[alive],
            K_r=self.cells.K_r[alive],
            v_max=self.cells.v_max[alive],
        )

        # distribute x1 to internal and external molecules
        self.culture.molecule_map[:, xs, ys] = x1[:, self._ext_mol_idxs].T
        self.cells.molecules[alive] = x1[:, self._int_mol_idxs]

        # age cells
        self.cells.ages += self.time_step

    def diffuse_molecules(self) -> None:
        self.culture.diffuse_molecules()

    def mutate_cells(
        self,
        cell_idxs: list[int] | None = None,
        p: float = 1e-6,
        p_indel: float = 0.4,
        p_del: float = 0.66,
    ) -> None:
        idxs = cell_idxs or (self.cells.get_available_idxs().tolist())
        seqs = self.cells.genomes[idxs]
        mutated = point_mutations(seqs=seqs, p=p, p_indel=p_indel, p_del=p_del)
        pairs = [(d, idxs[i]) for d, i in mutated]
        self.update_cells(genome_idx_pairs=pairs)

    def recombinate_cells(
        self, cell_idxs: list[int] | None = None, p: float = 1e-7
    ) -> None:
        idxs = cell_idxs or (self.cells.get_available_idxs().tolist())
        nghbrs = self.get_neighbors(cell_idxs=idxs)
        genomes0 = self.cells.genomes[[d[0] for d in nghbrs]]
        genomes1 = self.cells.genomes[[d[1] for d in nghbrs]]
        pairs = [(g0, g1) for g0, g1 in zip(genomes0, genomes1)]
        mutated = recombinations(seq_pairs=pairs, p=p)

        genome_idx_pairs = []
        for c0, c1, idx in mutated:
            c0_i, c1_i = nghbrs[idx]
            genome_idx_pairs.append((c0, c0_i))
            genome_idx_pairs.append((c1, c1_i))

        self.update_cells(genome_idx_pairs=genome_idx_pairs)

    def save(self, rundir: Path, name: str = "world.pkl") -> None:
        rundir.mkdir(parents=True, exist_ok=True)
        with open(rundir / name, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def from_file(
        self,
        rundir: Path,
        name: str = "world.pkl",
        device: str | None = None,
    ) -> "World":
        with open(rundir / name, "rb") as fh:
            unpickler = _CPU_Unpickler(fh, map_location=device)
            obj: World = unpickler.load()

        if device is not None:
            obj.device = device
            obj.kinetics.device = device

        return obj

    def save_state(self, statedir: Path):
        statedir.mkdir(parents=True, exist_ok=True)
        self.culture.save_state(statedir=statedir)
        self.cells.save_state(statedir=statedir)

    def load_state(self, statedir: Path):
        self.culture.load_state(statedir=statedir)
        self.cells.load_state(statedir=statedir)
