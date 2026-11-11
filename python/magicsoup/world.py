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
from magicsoup.core.tensorclass import T, TensorClass, TensorKwargs
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


# TODO: adapt torch code:
# convert IDXDTYPE to long (pre-requisite)
# V[idx], 1-D index on dim 0        ->      V.index_select(0, idx)
# V[idx], 2-D+ index along another axis     ->      V.gather(dim, idx)
# a[idx] = b        ->      a.index_copy_(0, idx, b)
# a[idx] += b       ->      a.index_put_((idx,), b, accumulate=True) or scatter_add_
# a[mask] = b[mask]     ->      torch.where(mask, b, a, out=a)
# prio dim-0 cases, hot path


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


def _iter_tensorclasses(root: object):
    """Yield every [TensorClass][magicsoup.core.tensorclass.TensorClass] reachable from `root` (incl. `root`).

    Descends only through `TensorClass` instances; within a `World` they all
    reference each other (`World` -> `cells`/`culture`/..., `cells` -> `proteomics`
    -> maps, `culture` -> neighbourhood), so this reaches every one.
    """
    seen: set[int] = set()
    stack: list[object] = [root]
    while stack:
        obj = stack.pop()
        if not isinstance(obj, TensorClass) or id(obj) in seen:
            continue
        seen.add(id(obj))
        yield obj
        stack.extend(vars(obj).values())


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
        # forces host sync
        idx = -1

        if by_idx is not None:
            if by_idx >= self.cells.c or not bool(self.cells.alive[by_idx]):
                raise ValueError(f"Cell with idx {by_idx} not found")
            idx = by_idx

        if by_position is not None and by_position != (-1, -1):
            cell_coords = self.culture.get_cell_coords()
            mask = (cell_coords == self.idxtensor([by_position])).all(dim=1)
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

    def spawn_cells(self, genomes: list[str]) -> T:
        # forces host sync
        n_new_cells = len(genomes)
        if n_new_cells > self.culture.N2:
            raise ValueError(
                f"Wanted to spawn {n_new_cells:,} new cells"
                f" but there are only {self.culture.N2:,} voxels."
            )
        if n_new_cells == 0:
            return self.idxtensor([])

        n_avail_pos = int((self.culture.cell_map < 0).sum())
        if n_avail_pos == 0:
            _log.warning("No free positions to spawn %d cells", n_new_cells)
            return self.idxtensor([])

        if n_avail_pos < n_new_cells:
            _log.warning("Only %d free positions to spawn %d cells", n_avail_pos, n_new_cells)
            n_new_cells = n_avail_pos
            random.shuffle(genomes)
            genomes = genomes[:n_new_cells]

        # adjust c if necessary (sync with host)
        n_cells = self.cells.get_n_cells()
        self.cells.update_c(c_req=n_cells + n_new_cells)

        # get random target voxels
        mask = self.cells.get_free_mask(n=n_new_cells)  # (c,)
        voxels = self.culture.get_random_free_voxels(mask=mask)  # (c,)

        # place the selected cells on their target voxels
        self.culture.place_cells(voxels=voxels)

        # cell is picking up the molecules of the pxl it is born on
        # it receives the same concentration, total volume increases
        coords = self.culture.voxels_to_coords(voxels)
        rows = coords[:, 0]
        cols = coords[:, 1]
        f = self.v_voxel / (self.v_voxel + self.v_cell)
        pickup = self.culture.molecule_map[:, rows, cols]
        self.cells.molecules = torch.where(mask[:, None], pickup.T * f, self.cells.molecules)
        self.culture.molecule_map[:, rows, cols] = pickup * f

        new_idxs = mask.nonzero_static(size=n_new_cells, fill_value=-1).to(self.idxtype).flatten()

        # update genomes and proteomes
        self.cells.update_genomes(genomes=StrArray.from_strings(genomes), idxs=new_idxs)

        # update other values
        self.cells.alive[new_idxs] = True
        self.cells.ages[new_idxs] = 0.0
        self.cells.generations[new_idxs] = 0
        self.cells.lineages[new_idxs.cpu().numpy()] = [randstr(n=12) for _ in range(n_new_cells)]

        return new_idxs

    def add_cells(self, cells: list["Cell"]) -> T:
        # forces host sync
        n_new_cells = len(cells)
        if n_new_cells > self.culture.N2:
            raise ValueError(
                f"Wanted to add {n_new_cells:,} new cells"
                f" but there are only {self.culture.N2:,} voxels."
            )
        if n_new_cells == 0:
            return self.idxtensor([])

        n_avail_pos = int((self.culture.cell_map < 0).sum())
        if n_avail_pos == 0:
            _log.warning("No free voxels to add %d cells", n_new_cells)
            return self.idxtensor([])

        if n_avail_pos < n_new_cells:
            _log.warning("Only %d free voxels to add %d cells", n_avail_pos, n_new_cells)
            n_new_cells = n_avail_pos
            random.shuffle(cells)
            cells = cells[:n_new_cells]

        # adjust c if necessary (sync with host)
        n_cells = self.cells.get_n_cells()
        self.cells.update_c(c_req=n_cells + n_new_cells)

        # get random target voxels
        mask = self.cells.get_free_mask(n=n_new_cells)  # (c,)
        voxels = self.culture.get_random_free_voxels(mask=mask)

        # place the selected cells on their target voxels
        self.culture.place_cells(voxels=voxels)

        new_idxs = mask.nonzero_static(size=n_new_cells, fill_value=-1).to(self.idxtype).flatten()

        # update parameters from genomes (sync with host)
        genomes = StrArray.from_strings([d.genome for d in cells])
        self.cells.update_genomes(genomes=genomes, idxs=new_idxs)

        # update other parameters
        self.cells.alive[new_idxs] = True
        self.cells.lineages[new_idxs.cpu().numpy()] = [d.lineage for d in cells]
        self.cells.molecules[new_idxs, :] = torch.stack([d.molecules for d in cells])
        self.cells.ages[new_idxs] = self.ftensor([d.age for d in cells])
        self.cells.generations[new_idxs] = self.idxtensor([d.generation for d in cells])

        return new_idxs

    def resuspend_cells(self, mask: T | None = None) -> None:
        """
        Relocate cells to random free voxels

        Arguments:
            mask: (c,) bool tensor of cells to relocate; None = all living

        Positions are updated in place. Each selected cell is placed on a distinct,
        randomly chosen free voxel (its own vacated voxel is eligible).
        """
        if mask is None:
            mask = self.cells.alive.clone()
        else:
            mask = mask & self.cells.alive

        # unoccupy cell map
        nn_idx = self.culture.cell_map.clamp(min=0)  # (N, N) -1's -> 0
        nn_rm = (self.culture.cell_map >= 0) & mask[nn_idx]  # (N, N)
        self.culture.cell_map.masked_fill_(nn_rm, -1)

        voxels = self.culture.get_random_free_voxels(mask=mask)
        self.culture.place_cells(voxels=voxels)

    def migrate_cells(self, mask: T | None = None) -> T:
        """
        Let cells migrate to a free voxel in their Moore neighbourhood.

        Arguments:
            mask: (c,) bool tensor of cells attempting to migrate; None = all living

        Returns:
            (c,) bool tensor of cells that successfully migrated

        Positions are updated in place.
        """
        if mask is None:
            mask = self.cells.alive.clone()
        else:
            mask = mask & self.cells.alive

        # positions are updated in place
        return self.culture.migrate_cells_randomly(mask=mask)

    def divide_cells(self, mask: T | None = None) -> T:
        """
        Divide cells, placing each child in a free voxel of its parent.

        Arguments:
            mask: (c,) bool tensor of parents that want to divide; None = all living

        Returns:
            (c,) index tensor `parent_of` where `parent_of[s]` is the parent slot
            that produced the child now occupying slot `s`, else -1.

        Grows the `c` dimension with hysteresis if there are not enough free slots.
        Children inherit their parent's proteome and half of its molecules.
        Forces host synchronization.
        """
        cells = self.cells
        culture = self.culture
        if mask is None:
            mask = cells.alive.clone()
        else:
            mask = mask & cells.alive

        # resolve which parents win a free neighbour voxel (reserves those voxels)
        success, won_voxel = culture.divide_cells(mask=mask)  # (c,), (c,)

        # grow c to fit the new children (host sync)
        c_req = int(cells.alive.sum() + success.sum())  # sync with host
        c = cells.update_c(c_req=c_req)

        # pad to the (possibly grown) c; appended slots are not parents
        success = self._pad_to_c(success, c, False)
        won_voxel = self._pad_to_c(won_voxel, c, -1)

        # allocate a free child slot per successful parent
        child_of = cells.allocate_children(divided=success)
        parent_of = self.invert_idxmap(child_of)

        # copy parent params to their children (sync with host)
        cells.update_descendants(parent_of=parent_of)
        cells.copy_proteomes(idx_map=parent_of)

        # replace reservations with the real child ids
        voxels = self.remap_gather(tgt=won_voxel, src=parent_of)
        self.culture.place_cells(voxels=voxels)

        return parent_of

    def kill_cells(self, mask: T | None = None) -> None:
        """
        Kill cells, spilling their molecules onto their voxels.

        Arguments:
            mask: (c,) bool tensor of cells to kill; None = all living

        To avoid host synchronization dead cells' genomes and lineages are not cleared
        from `world.cells`. If you want to remove them use [compact()][.compact].
        """
        cells = self.cells
        culture = self.culture
        if mask is None:
            mask = cells.alive.clone()
        else:
            mask = mask & cells.alive  # only living cells can be killed

        # per-voxel kill mask
        nn_idx = culture.cell_map.clamp(min=0)  # (N, N) -1 -> 0
        nn_kill = (culture.cell_map >= 0) & mask[nn_idx]  # (N, N)

        # spill killed cells' molecules onto their voxels
        mols = cells.molecules[nn_idx].permute(2, 0, 1)  # (m, N, N)
        spill = torch.where(nn_kill, mols, self.ftensor(0.0))  # (m, N, N)
        culture.molecule_map += spill * (self.v_cell / self.v_voxel)

        # free up cell map
        culture.cell_map.masked_fill_(nn_kill, -1)

        # unset device cell state
        cells.reset_cells(mask=mask)
        cells.reset_proteomes(mask=mask)

    def mutate_cells(
        self,
        mask: T | None = None,
        mutator: Mutator | None = None,
    ) -> None:
        """
        Mutate the genomes of the selected living cells.

        Arguments:
            mask: (c,) bool tensor of cells to mutate; None = all living cells
            mutator: to override the word's default [Mutator][magicsoup.mutations.Mutator]

        Clears dead cells' genomes first (see
        [reconcile()][magicsoup.world.World.reconcile]), then mutates the selected
        living cells and updates their proteomes. Independent of division: pass a
        mask to restrict mutation (e.g. only new children) or leave it to mutate
        the whole population.
        """
        # forces host sync
        # clear dead genomes so the arena stays compact and dead cells aren't hit
        alive_host = self.cells.reconcile_host()
        self.cells.genomes.compact()
        self.cells.lineages.compact()

        sel_idxs: np.ndarray
        if mask is None:
            sel_idxs = np.nonzero(alive_host)[0]
        else:
            sel_idxs = (mask & self.cells.alive).nonzero(as_tuple=True)[0].cpu().numpy()
        if sel_idxs.shape[0] == 0:
            return

        mutator = mutator or self.mutator
        idxs, mutated = mutator.mutate(genomes=self.cells.genomes[sel_idxs])

        if len(mutated) == 0:
            return

        self.cells.update_genomes(
            genomes=mutated, idxs=torch.from_numpy(sel_idxs[idxs]).to(self.device)
        )

    def advance_time(self, n_steps: int = 1) -> None:
        # forces host sync
        cell_idxs = self.cells.get_cell_idxs()  # (c,)

        cell_molecules = self.cells.molecules  # (c, m)
        molecule_map = self.culture.molecule_map  # (m, N, N)
        cell_coords = self.culture.get_cell_coords()  # (N², 2)
        cell_rows = cell_coords[cell_idxs, 0]
        cell_cols = cell_coords[cell_idxs, 1]

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
                [cell_molecules[cell_idxs], molecule_map[:, cell_rows, cell_cols].T],
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
            molecule_map[:, cell_rows, cell_cols] = x[:, self._ext_mol_idxs].T
            cell_molecules[cell_idxs] = x[:, self._int_mol_idxs]

        # age cells
        self.cells.ages += self.time_step

    def compact(self, util_lt: float | None = None) -> None:
        """Reclaim surplus cell slots and protein columns

        Arguments:
            util_lt: If defined, get current utilization (host synchronization)
                and only compact if current utilization is less than `util_lt`.

        Reconciles device with host and re-shapes tensors on device to cover
        current number of cells (`c` dimension) and maximum number of proteins (`p` dimension)
        more effectively. Call it periodically (e.g. after subsampling), not inside a hot step.

        With default scaling dimension (`dim_scaling` in `Cells`) the hysteresis during normal
        growth conditions creates a maximum utilization of about 70%. So, if `util_lt` is used
        it should be considerably lower (e.g. 50%) to avoid frequent re-shaping.
        """
        if util_lt is not None:
            # forces host sync
            util = self.cells.get_utilization()
            if util >= util_lt:
                return

        # forces host sync
        self.cells.reconcile_host()
        old_to_new = self.cells.compact()
        self.culture.update_cell_map(old_to_new)

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
            # map_location relocated the tensors; realign every sub-object's cached
            # device so later tensor creation (e.g. self.fzeros) lands on the new one
            for tc in _iter_tensorclasses(obj):
                tc.device = device

        return obj

    def save_state(self, statedir: Path):
        statedir.mkdir(parents=True, exist_ok=True)
        self.culture.save_state(statedir=statedir)
        self.cells.save_state(statedir=statedir)

    def load_state(self, statedir: Path):
        self.culture.load_state(statedir=statedir)
        self.cells.load_state(statedir=statedir)

    def _pad_to_c(self, t: T, c: int, fill) -> T:
        n = t.shape[0]
        if n >= c:
            return t[:c]
        pad = torch.full((c - n,), fill, dtype=t.dtype, device=t.device)
        return torch.cat([t, pad])
