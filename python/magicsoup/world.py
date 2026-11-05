import logging
import pickle
import random
from io import BytesIO
from pathlib import Path

import numpy as np
import torch

from magicsoup.biology import Cell
from magicsoup.cells import Cells, CellsKwargs
from magicsoup.chemistry import Chemistry
from magicsoup.core.strarray import StrArray
from magicsoup.core.tensorclass import TensorClass, TensorKwargs
from magicsoup.core.util import randstr
from magicsoup.culture import Culture, CultureKwargs
from magicsoup.genomics import Genomics, GenomicsKwargs
from magicsoup.kinetics import Kinetics, KineticsKwargs
from magicsoup.mutations import Mutator, MutatorKwargs
from magicsoup.physics import (
    Degradation,
    Diffusion,
    DiffusionKwargs,
    Permeation,
    PermeationKwargs,
)
from magicsoup.proteomics import Proteomics, ProteomicsKwargs

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


# TODO: getting the current cell idxs as tensor always synchronizes
#       I could add methods which are applied using a bool tensor
#       or which are applied for all cells
#       this could then be implemented without host synchronization


class World(TensorClass):

    def __init__(
        self,
        chemistry: Chemistry,
        genomics: Genomics | None = None,
        mutator: Mutator | None = None,
        proteomics: Proteomics | None = None,
        kinetics: Kinetics | None = None,
        diffusion: Diffusion | None = None,
        permeation: Permeation | None = None,
        degradation: Degradation | None = None,
        culture: Culture | None = None,
        cells: Cells | None = None,
        time_step: float = 1.0,  # s
        v_cell: float = 1.0,
        v_voxel: float = 1000.0,
        kinetics_kwargs: KineticsKwargs | None = None,
        diffusion_kwargs: DiffusionKwargs | None = None,
        permeation_kwargs: PermeationKwargs | None = None,
        culture_kwargs: CultureKwargs | None = None,
        cells_kwargs: CellsKwargs | None = None,
        genomics_kwargs: GenomicsKwargs | None = None,
        mutator_kwargs: MutatorKwargs | None = None,
        proteomics_kwargs: ProteomicsKwargs | None = None,
        tensor_kwargs: TensorKwargs | None = None,
    ):
        super().__init__(**(tensor_kwargs or {}))

        self.time_step = time_step
        self.v_cell = v_cell
        self.v_voxel = v_voxel

        self.chemistry = chemistry
        self.culture = culture or (
            Culture(
                **(culture_kwargs or {}),
                chemistry=chemistry,
                tensor_kwargs=tensor_kwargs,
            )
        )
        self.cells = cells or Cells(
            **(cells_kwargs or {}),
            c_max=self.culture.map_size**2,
            chemistry=chemistry,
            genomics=genomics,
            proteomics=proteomics,
            genomics_kwargs=genomics_kwargs,
            proteomics_kwargs=proteomics_kwargs,
            tensor_kwargs=tensor_kwargs,
        )
        self.kinetics = kinetics or Kinetics(
            **(kinetics_kwargs or {}),
            dt=time_step,
            tensor_kwargs=tensor_kwargs,
        )
        self.diffusion = diffusion or Diffusion(
            **(diffusion_kwargs or {}),
            diff_coeffs=[d.diffusivity for d in chemistry.molecules],
            N=self.culture.map_size,
            dt=time_step,
            tensor_kwargs=tensor_kwargs,
        )
        self.permeation = permeation or Permeation(
            **(permeation_kwargs or {}),
            perm_coeffs=[d.permeability for d in chemistry.molecules],
            v_cell=v_cell,
            v_vox=v_voxel,
            dt=time_step,
            tensor_kwargs=tensor_kwargs,
        )
        self.degradation = degradation or Degradation(
            half_lifes=[d.half_life for d in chemistry.molecules],
            dt=time_step,
            tensor_kwargs=tensor_kwargs,
        )
        self.mutator = mutator or Mutator(**(mutator_kwargs or {}))

        n_molecules = len(chemistry.molecules)
        self._int_mol_idxs = list(range(n_molecules))
        self._ext_mol_idxs = list(range(n_molecules, n_molecules * 2))

    def get_cell(
        self,
        by_idx: int | None = None,
        by_position: tuple[int, int] | None = None,
    ) -> "Cell":
        # TODO: weird side effect of a getter
        self.reconcile()  # ensure the genome/lineage arrays are current
        idx = -1
        positions = self.culture.get_cell_positions()

        if by_idx is not None:
            idxs = np.where(self.cells.get_cell_idxs() == by_idx)[0]
            if idxs.shape[0] == 0:
                raise ValueError(f"Cell with idx {by_idx} not found")
            idx = by_idx

        if by_position is not None and by_position != (-1, -1):
            cellpos = self.idxtensor([by_position])
            positions = self.culture.get_cell_positions()
            mask = (positions == cellpos).all(dim=1)
            if not mask.any():
                raise ValueError(f"Cell at {by_position} not found")
            idx = int(torch.argwhere(mask)[0].item())

        genomes = self.cells.genomes[np.array([idx])]
        proteome_specs = self.cells.genomics.translate_genomes(genomes=genomes)
        proteome = (
            self.cells.proteomics.get_proteome(proteome=proteome_specs[0])
            if len(proteome_specs[0]) > 0
            else []
        )

        return Cell(
            genome=genomes[0],
            proteome=proteome,
            lineage=str(self.cells.lineages[idx]),
            age=float(self.cells.ages[idx]),
            generation=int(self.cells.generations[idx]),
            molecules=self.cells.molecules[idx, :],
        )

    def spawn_cells(self, genomes: list[str]) -> torch.Tensor:
        n_new_cells = len(genomes)
        if n_new_cells > self.culture.N2:
            raise ValueError(
                f"Wanted to spawn {n_new_cells:,} new cells"
                f" but there are only {self.culture.N2:,} voxels."
            )
        if n_new_cells == 0:
            return self.idxtensor([])

        free_pos = self.culture.get_free_positions(n=n_new_cells)
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

        n_living_cells = self.cells.get_n_cells()
        self.cells.update_c(c_req=n_living_cells + n_new_cells)
        free_idxs = self.cells.get_free_idxs()
        new_idxs = free_idxs[:n_new_cells]

        # occupy positions
        new_pos = free_pos[:n_new_cells]
        xs = new_pos[:, 0]
        ys = new_pos[:, 1]
        self.culture.cell_map[xs, ys] = new_idxs

        # cell is picking up the molecules of the pxl it is born on
        # it receives the same concentration, total volume increases
        f = self.v_voxel / (self.v_voxel + self.v_cell)
        pickup = self.culture.molecule_map[:, xs, ys]
        self.cells.molecules[new_idxs, :] = pickup.T * f
        self.culture.molecule_map[:, xs, ys] = pickup * f

        # update genomes and proteomes
        self.cells.update_genomes(genomes=StrArray.from_strings(genomes), idxs=new_idxs)

        # update other values
        self.cells.alive[new_idxs] = True
        self.cells.ages[new_idxs] = 0.0
        self.cells.generations[new_idxs] = 0
        self.cells.lineages[new_idxs.cpu().numpy()] = [
            randstr(n=12) for _ in range(n_new_cells)
        ]

        return new_idxs

    def add_cells(self, cells: list["Cell"]) -> torch.Tensor:
        n_new_cells = len(cells)
        if n_new_cells > self.culture.N2:
            raise ValueError(
                f"Wanted to add {n_new_cells:,} new cells"
                f" but there are only {self.culture.N2:,} voxels."
            )
        if n_new_cells == 0:
            return self.idxtensor([])

        free_pos = self.culture.get_free_positions(n=n_new_cells)
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

        n_cells = self.cells.get_n_cells()  # sync with host
        self.cells.update_c(c_req=n_cells + n_new_cells)
        free_idxs = self.cells.get_free_idxs()  # sync with host
        new_idxs = free_idxs[:n_new_cells]

        # occupy positions
        new_pos = free_pos[:n_new_cells]
        xs = new_pos[:, 0]
        ys = new_pos[:, 1]
        self.culture.cell_map[xs, ys] = new_idxs

        # update parameters from genomes
        genomes = StrArray.from_strings([d.genome for d in cells])
        self.cells.update_genomes(genomes=genomes, idxs=new_idxs)

        # update other parameters
        self.cells.alive[new_idxs] = True
        self.cells.lineages[new_idxs.cpu().numpy()] = [d.lineage for d in cells]
        self.cells.molecules[new_idxs, :] = torch.stack([d.molecules for d in cells])
        self.cells.ages[new_idxs] = self.ftensor([d.age for d in cells])
        self.cells.generations[new_idxs] = self.idxtensor([d.generation for d in cells])

        return new_idxs

    def resuspend_cells(self, mask: torch.Tensor | None = None) -> None:
        """
        Relocate cells to random free voxels (e.g. to model a mixed suspension).

        Arguments:
            mask: (c,) bool tensor of cells to relocate; None = all living

        Positions are updated in place. Each selected cell is placed on a distinct,
        randomly chosen free voxel (its own vacated voxel is eligible). Operates
        over all cell slots using masks and performs no host synchronization.
        """
        cells = self.cells
        culture = self.culture
        if mask is None:
            mask = cells.alive.clone()
        else:
            mask = mask & cells.alive

        n2 = culture.N2
        c = mask.shape[0]
        coords = culture.get_cell_positions()[:c]  # (c, 2) current positions
        old_flat = culture._positions_to_idxs(coords)  # (c,) current voxel
        cell_ids = self.idxarange(c)  # (c,)
        flat = culture.cell_map_flat

        # vacate the selected cells' voxels (non-selected routed to a dump slot)
        padded = torch.cat([flat, self.idxtensor([-1])])  # (N²+1,)
        vacate = torch.where(mask, old_flat, self.idxtensor(n2))  # (c,)
        padded.scatter_(0, vacate, self.idxfull(c, value=-1))
        flat[:] = padded[:n2]

        # rank now-free voxels in random order; assign the r-th selected cell the
        # r-th free voxel (there are always >= as many free voxels as selected
        # cells, since each selected cell just vacated one)
        free = flat < 0  # (N²,)
        scores = self.frand(n2).masked_fill(~free, 2.0)  # free in [0, 1), occupied 2
        order = scores.argsort()  # (N²,) free voxels first, in random order
        sel_rank = mask.cumsum(0, dtype=self.idxtype) - 1  # (c,)
        target = order[sel_rank.clamp(min=0)].int()  # (c,) target voxel per cell

        # place the selected cells on their target voxels
        padded = torch.cat([flat, self.idxtensor([-1])])
        place = torch.where(mask, target, self.idxtensor(n2))  # (c,)
        padded.scatter_(0, place, cell_ids)
        flat[:] = padded[:n2]

    def migrate_cells(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Let cells migrate to a free voxel in their Moore neighbourhood.

        Arguments:
            mask: (c,) bool tensor of cells attempting to migrate; None = all living

        Returns:
            (c,) bool tensor of cells that successfully migrated

        Positions are updated in place. Performs no host synchronization.
        """
        if mask is None:
            mask = self.cells.alive.clone()
        else:
            mask = mask & self.cells.alive

        # positions are updated in place
        return self.culture.undirected_cell_migration(mask=mask)

    def divide_cells(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Divide cells, placing each child in a free voxel of its parent.

        Arguments:
            mask: (c,) bool tensor of parents that want to divide; None = all living

        Returns:
            (c,) index tensor `parent_of` where `parent_of[s]` is the parent slot
            that produced the child now occupying slot `s`, else -1.

        Grows the `c` dimension with hysteresis if there are not enough free slots.
        Children inherit their parent's proteome and half of its molecules.
        """
        cells = self.cells
        culture = self.culture
        if mask is None:
            mask = cells.alive.clone()
        else:
            mask = mask & cells.alive

        # resolve which parents win a free neighbour voxel (reserves those voxels)
        success, won_voxel = culture.cell_division(mask=mask)  # (c,), (c,)

        # grow c to fit the new children (the one host sync)
        c_req = int(cells.alive.sum() + success.sum())  # sync with host
        cells.update_c(c_req=c_req)
        c = cells.c

        # pad to the (possibly grown) c; appended slots are not parents
        success = self._pad_to_c(success, c, False)
        won_voxel = self._pad_to_c(won_voxel, c, -1)

        # allocate a free child slot per successful parent (prefix-sum, device)
        parent_of, child_slot = self._allocate_children(success)

        # copy children's genomes and lineages from their parents on the host
        # (adjacent to the grow sync above, so ~one combined stall)
        po = parent_of.cpu().numpy()  # sync with host
        is_child = po >= 0
        parents = po[is_child]
        cells.genomes[is_child] = cells.genomes[parents]
        cells.lineages[is_child] = cells.lineages[parents]

        # finalize the map: replace reservations with the real child ids
        dump = culture.N2
        place = torch.where(success, won_voxel, self.idxtensor(dump))  # (c,)
        child_vals = torch.where(success, child_slot, self.idxtensor(0))  # (c,)
        flat = culture.cell_map_flat
        padded = torch.cat([flat, self.idxtensor([-1])])
        padded.scatter_(0, place, child_vals)
        flat[:] = padded[: culture.N2]

        # copy proteomes and cell state from parents to their children (device)
        cells.copy_proteomes_masked(parent_of=parent_of)
        self._apply_division_state(parent_of=parent_of, divided=success)

        return parent_of

    def _pad_to_c(self, t: torch.Tensor, c: int, fill) -> torch.Tensor:
        n = t.shape[0]
        if n >= c:
            return t[:c]
        pad = torch.full((c - n,), fill, dtype=t.dtype, device=t.device)
        return torch.cat([t, pad])

    def _allocate_children(
        self, success: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # assign the i-th successful parent to the i-th free cell slot, sync-free
        cells = self.cells
        c = success.shape[0]
        free = ~cells.alive  # (c,)
        order = self.idxarange(c)  # (c,)

        # map "r-th free slot" -> its cell index (non-free rows routed to dump c)
        rank_of_free = torch.where(
            free, free.cumsum(0, dtype=self.idxtype) - 1, self.idxtensor(c)
        )  # (c,)
        slot_of_rank = order.new_full((c + 1,), -1)
        slot_of_rank.scatter_(0, rank_of_free, order)

        # each successful parent -> next free slot
        prank = success.cumsum(0, dtype=self.idxtype) - 1  # (c,)
        child_slot = torch.where(
            success, slot_of_rank[prank.clamp(min=0)], self.idxtensor(-1)
        )  # (c,)

        # invert to parent_of[child_slot] = parent
        cs = torch.where(success, child_slot, self.idxtensor(c))  # non-succ -> dump
        parent_of = self.idxfull(c + 1, value=-1)
        parent_of.scatter_(0, cs, order)
        return parent_of[:c], child_slot

    def _apply_division_state(
        self, parent_of: torch.Tensor, divided: torch.Tensor
    ) -> None:
        cells = self.cells
        is_child = parent_of >= 0  # (c,)
        src = parent_of.clamp(min=0)  # (c,)
        descendant = is_child | divided  # (c,) children + parents that divided

        # children become alive
        cells.alive = cells.alive | is_child
        # children inherit parent generation; all descendants then increment
        cells.generations = torch.where(
            is_child, cells.generations[src], cells.generations
        )
        cells.generations = cells.generations + descendant.to(cells.generations.dtype)
        # descendants reset age
        cells.ages = torch.where(descendant, self.ftensor(0.0), cells.ages)
        # children share parent molecules, then all descendants split in half
        cells.molecules = torch.where(
            is_child[:, None], cells.molecules[src], cells.molecules
        )
        cells.molecules = torch.where(
            descendant[:, None], cells.molecules * 0.5, cells.molecules
        )

    def compact(self) -> None:
        """
        Reclaim surplus cell slots and protein columns (with hysteresis).

        Shrinks the `c` and `p` dimensions of all cell tensors to the current
        occupancy, reindexes cells, and remaps the cell map so that cells keep
        their positions. Synchronizes with the host; call periodically
        (e.g. after subsampling), not inside the hot step.
        """
        new_idx = self.cells.compact()  # (c_old,) old -> new cell index (-1 dropped)
        flat = self.culture.cell_map_flat
        flat[:] = torch.where(flat >= 0, new_idx[flat.clamp(min=0)], flat)

    def reconcile(self) -> None:
        """
        Clear dead cells' genomes and lineages (one host synchronization).

        Kill defers this host-side cleanup to keep itself sync-free. This is
        called automatically by [mutate_cells()][magicsoup.world.World.mutate_cells],
        [save()][magicsoup.world.World.save],
        [save_state()][magicsoup.world.World.save_state] and
        [get_cell()][magicsoup.world.World.get_cell]; call it manually if you read
        the genome/lineage arrays directly after killing cells.
        """
        self.cells.reconcile_host()

    def update_cells(self, genomes: StrArray, cell_idxs: torch.Tensor) -> None:
        if len(genomes) == 0:
            return

        if len(genomes) != cell_idxs.numel():
            raise ValueError("Length of genomes of cell_idxs doesn't match")

        self.cells.update_genomes(genomes=genomes, idxs=cell_idxs)

    def kill_cells(self, mask: torch.Tensor | None = None) -> None:
        """
        Kill cells, spilling their molecules onto their voxels.

        Arguments:
            mask: (c,) bool tensor of cells to kill; None = all living

        Device-only: all cell tensors and the cell map are updated with masks.
        Dead cells' genomes and lineages are cleared at the next
        [reconcile()][magicsoup.world.World.reconcile], so this performs no host
        synchronization.
        """
        cells = self.cells
        culture = self.culture
        if mask is None:
            mask = cells.alive.clone()
        else:
            mask = mask & cells.alive  # only living cells can be killed

        # per-voxel kill mask by gathering each voxel's occupant's kill status
        cm = culture.cell_map  # (N, N)
        safe = cm.clamp(min=0)  # (N, N) -1 (empty) -> 0
        kill_here = (cm >= 0) & mask[safe]  # (N, N)

        # spill killed cells' molecules onto their voxels, then free the map
        mols = cells.molecules[safe].permute(2, 0, 1)  # (m, N, N) per-voxel cell mols
        spill = torch.where(kill_here, mols, self.ftensor(0.0))  # (m, N, N)
        culture.molecule_map += spill * (self.v_cell / self.v_voxel)
        cm.masked_fill_(kill_here, -1)

        # unset device cell state (genomes/lineages deferred to reconcile)
        m1 = mask[:, None]
        m2 = mask[:, None, None]
        cells.alive = cells.alive & ~mask
        cells.ages = torch.where(mask, self.ftensor(0.0), cells.ages)
        cells.generations = torch.where(mask, self.idxtensor(0), cells.generations)
        cells.molecules = torch.where(m1, self.ftensor(0.0), cells.molecules)
        cells.v_max = torch.where(m1, self.ftensor(0.0), cells.v_max)
        cells.k_f = torch.where(m1, self.ftensor(0.0), cells.k_f)
        cells.k_b = torch.where(m1, self.ftensor(0.0), cells.k_b)
        cells.k_r = torch.where(m1, self.ftensor(0.0), cells.k_r)
        cells.N_f = torch.where(m2, self.itensor(0), cells.N_f)
        cells.N_b = torch.where(m2, self.itensor(0), cells.N_b)
        cells.N_h = torch.where(m2, self.itensor(0), cells.N_h)
        cells.N_e = torch.where(m2, self.ftensor(0.0), cells.N_e)

    def advance_time(self, n_steps: int = 1) -> None:
        cell_molecules = self.cells.molecules  # (c, m)
        cell_idxs = self.cells.get_cell_idxs()  # (c,) sync with host
        molecule_map = self.culture.molecule_map  # (m, N, N)
        cell_positions = self.culture.get_cell_positions()  # (N², 2)
        xs = cell_positions[cell_idxs, 0]
        ys = cell_positions[cell_idxs, 1]

        if cell_idxs.numel() == 0:
            return

        for _ in range(n_steps):
            # degradation
            molecule_map[:] = self.degradation.degrade(x=molecule_map, m_dim=0)
            cell_molecules[:] = self.degradation.degrade(x=cell_molecules, m_dim=1)

            # diffusion
            molecule_map[:] = self.diffusion.diffuse(molecule_map)

            # collect internal and external molecules for x
            x = torch.cat(
                [cell_molecules[cell_idxs], molecule_map[:, xs, ys].T],
                dim=1,
            )

            # permeation
            x = self.permeation.permeate(x)

            # enzymatic activity
            x = self.kinetics.get_x1(
                x0=x,
                N_e=self.cells.N_e[cell_idxs],
                N_f=self.cells.N_f[cell_idxs],
                N_b=self.cells.N_b[cell_idxs],
                N_h=self.cells.N_h[cell_idxs],
                k_f=self.cells.k_f[cell_idxs],
                k_b=self.cells.k_b[cell_idxs],
                k_r=self.cells.k_r[cell_idxs],
                v_max=self.cells.v_max[cell_idxs],
            )

            # distribute x1 to internal and external molecules
            molecule_map[:, xs, ys] = x[:, self._ext_mol_idxs].T
            cell_molecules[cell_idxs] = x[:, self._int_mol_idxs]

        # age cells
        self.cells.ages += self.time_step

    def mutate_cells(
        self,
        mask: torch.Tensor | None = None,
        mutator: Mutator | None = None,
    ) -> None:
        """
        Mutate the genomes of the selected living cells.

        Arguments:
            mask: (c,) bool tensor of cells to mutate; None = all living cells
            mutator: [Mutator][magicsoup.mutations.Mutator] to use; defaults to the
                world's own.

        Clears dead cells' genomes first (see
        [reconcile()][magicsoup.world.World.reconcile]), then mutates the selected
        living cells and updates their proteomes. Independent of division: pass a
        mask to restrict mutation (e.g. only new children) or leave it to mutate
        the whole population.
        """
        # clear dead genomes so the arena stays compact and dead cells aren't hit
        alive_host = self.cells.reconcile_host()  # sync with host
        if mask is None:
            sel_idxs = np.nonzero(alive_host)[0]
        else:
            # restrict to living cells, then pull the selection to the host
            sel_idxs = (mask & self.cells.alive).nonzero(as_tuple=True)[0].cpu().numpy()
        if sel_idxs.shape[0] == 0:
            return

        mutator = mutator or self.mutator
        idxs, mutated = mutator.mutate(genomes=self.cells.genomes[sel_idxs])

        self.update_cells(
            cell_idxs=torch.from_numpy(sel_idxs[idxs]).to(self.device),
            genomes=mutated,
        )

    def save(self, rundir: Path, name: str = "world.pkl") -> None:
        self.reconcile()  # bring host genome/lineage arrays in sync before saving
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
        self.reconcile()  # bring host genome/lineage arrays in sync before saving
        statedir.mkdir(parents=True, exist_ok=True)
        self.culture.save_state(statedir=statedir)
        self.cells.save_state(statedir=statedir)

    def load_state(self, statedir: Path):
        self.culture.load_state(statedir=statedir)
        self.cells.load_state(statedir=statedir)
