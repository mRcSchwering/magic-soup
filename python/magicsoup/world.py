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

    def resuspend_cells(self, cell_idxs: torch.Tensor | None = None) -> None:
        if cell_idxs is None:
            cell_idxs = self.cells.get_cell_idxs()  # sync with host

        if cell_idxs.numel() == 0:
            return

        n_cells = self.cells.get_n_cells()

        # unoccupy current positions
        cell_positions = self.culture.get_cell_positions()
        old_xs = cell_positions[cell_idxs, 0]
        old_ys = cell_positions[cell_idxs, 1]
        self.culture.cell_map[old_xs, old_ys] = -1

        # find new unoccupied positions
        new_pos = self.culture.get_free_positions(n=n_cells)
        new_xs = new_pos[cell_idxs, 0]
        new_ys = new_pos[cell_idxs, 1]

        self.culture.cell_map[new_xs, new_ys] = cell_idxs

    def migrate_cells(self, cell_idxs: torch.Tensor | None = None) -> torch.Tensor:
        if cell_idxs is None:
            cell_idxs = self.cells.get_cell_idxs()  # sync with host

        if cell_idxs.numel() == 0:
            return self.idxtensor([])

        # positions are updated in place
        migrated_idxs = self.culture.undirected_cell_migration(cell_idxs=cell_idxs)

        return migrated_idxs

    def divide_cells(
        self, cell_idxs: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cell_idxs is None:
            cell_idxs = self.cells.get_cell_idxs()  # sync with host

        n_req = cell_idxs.numel()
        if n_req == 0:
            return self.idxtensor([]), self.idxtensor([])

        # prepare for children
        avail_idxs = self.cells.get_free_idxs()  # sync with host
        n_avail = avail_idxs.numel()

        if n_avail >= n_req:
            child_idxs = avail_idxs[:n_req]
        else:
            n_alive = self.cells.get_n_cells()  # sync with host
            self.cells.update_c(c_req=n_req + n_alive)
            avail_idxs = self.cells.get_free_idxs()  # sync with host
            child_idxs = avail_idxs[:n_req]

        # already places children on cell map
        parent_idxs, child_idxs = self.culture.cell_division(
            parent_idxs=cell_idxs, child_idxs=child_idxs
        )

        n_new_cells = parent_idxs.numel()
        if n_new_cells == 0:
            return self.idxtensor([]), self.idxtensor([])

        # update child kinetics parameters from genomes
        self.cells.copy_proteomes(from_idxs=parent_idxs, to_idxs=child_idxs)
        cpu_child_idxs = child_idxs.cpu().numpy()
        cpu_parent_idxs = parent_idxs.cpu().numpy()

        # update cell parameters for children
        descendant_idxs = torch.cat([parent_idxs, child_idxs])
        self.cells.alive[child_idxs] = True
        self.cells.genomes[cpu_child_idxs] = self.cells.genomes[cpu_parent_idxs]
        self.cells.lineages[cpu_child_idxs] = self.cells.lineages[cpu_parent_idxs]
        self.cells.generations[child_idxs] = self.cells.generations[parent_idxs]
        self.cells.generations[descendant_idxs] += 1
        self.cells.ages[descendant_idxs] = 0.0

        # descendants share molecules
        self.cells.molecules[child_idxs] = self.cells.molecules[parent_idxs]
        self.cells.molecules[descendant_idxs] *= 0.5

        return parent_idxs, child_idxs

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

    def update_cells(self, genomes: StrArray, cell_idxs: torch.Tensor) -> None:
        if len(genomes) == 0:
            return

        if len(genomes) != cell_idxs.numel():
            raise ValueError("Length of genomes of cell_idxs doesn't match")

        self.cells.update_genomes(genomes=genomes, idxs=cell_idxs)

    def kill_cells(self, cell_idxs: torch.Tensor | None = None) -> None:
        if cell_idxs is None:
            cell_idxs = self.cells.get_cell_idxs()  # sync with host

        if cell_idxs.numel() == 0:
            return

        # sync with host
        cpu_cell_idxs = cell_idxs.cpu().numpy()

        # free up map
        cell_positions = self.culture.get_cell_positions()
        xs = cell_positions[cell_idxs, 0]
        ys = cell_positions[cell_idxs, 1]
        self.culture.cell_map[xs, ys] = -1

        # all molecules are spilled onto voxel
        n_voxel = self.culture.molecule_map[:, xs, ys] * self.v_voxel
        n_cell = self.cells.molecules[cell_idxs, :].T * self.v_cell
        self.culture.molecule_map[:, xs, ys] = (n_voxel + n_cell) / self.v_voxel

        # unset parameters
        self.cells.genomes[cpu_cell_idxs] = ""
        self.cells.lineages[cpu_cell_idxs] = ""
        self.cells.alive[cell_idxs] = False
        self.cells.ages[cell_idxs] = 0.0
        self.cells.generations[cell_idxs] = 0
        self.cells.molecules[cell_idxs] = 0.0

        self.cells.v_max[cell_idxs] = 0.0
        self.cells.k_f[cell_idxs] = 0.0
        self.cells.k_b[cell_idxs] = 0.0
        self.cells.k_r[cell_idxs] = 0.0
        self.cells.N_f[cell_idxs] = 0
        self.cells.N_b[cell_idxs] = 0
        self.cells.N_h[cell_idxs] = 0
        self.cells.N_e[cell_idxs] = 0

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
        self, cell_idxs: torch.Tensor | None = None, mutator: Mutator | None = None
    ) -> None:
        if cell_idxs is None:
            cell_idxs = self.cells.get_cell_idxs()  # sync with host

        if cell_idxs.numel() == 0:
            return

        cpu_cell_idxs = cell_idxs.cpu().numpy()
        mutator = mutator or self.mutator
        idxs, mutated = mutator.mutate(genomes=self.cells.genomes[cpu_cell_idxs])

        self.update_cells(
            cell_idxs=torch.from_numpy(cpu_cell_idxs[idxs]).to(self.device),
            genomes=mutated,
        )

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
