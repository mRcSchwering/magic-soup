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
from magicsoup.physics import (
    Degradation,
    Diffusion,
    DiffusionKwargs,
    Permeation,
    PermeationKwargs,
)
from magicsoup.proteomics import Proteomics, ProteomicsKwargs
from magicsoup.util import TensorClass, TensorKwargs, randstr

_log = logging.getLogger(__name__)


# TODO: @torch.inference_mode()
#       around major stepping functions may speed up things even a bit more
#       (or outside of the simulation around sep())
#       however it means every tensor created within inference_mode cannot be mutated
#       but this should be the case


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


class World(TensorClass):

    def __init__(
        self,
        chemistry: Chemistry,
        kinetics: Kinetics | None = None,
        diffusion: Diffusion | None = None,
        permeation: Permeation | None = None,
        degradation: Degradation | None = None,
        culture: Culture | None = None,
        cells: Cells | None = None,
        time_step: float = 1.0,  # s
        time_resolution: float = 0.1,  # s
        v_cell: float = 1.0,
        v_voxel: float = 1000.0,
        kinetics_kwargs: KineticsKwargs | None = None,
        diffusion_kwargs: DiffusionKwargs | None = None,
        permeation_kwargs: PermeationKwargs | None = None,
        culture_kwargs: CultureKwargs | None = None,
        cells_kwargs: CellsKwargs | None = None,
        genomics_kwargs: GenomicsKwargs | None = None,
        proteomics_kwargs: ProteomicsKwargs | None = None,
        tensor_kwargs: TensorKwargs | None = None,
    ):
        super().__init__(**(tensor_kwargs or {}))

        if not (time_step / time_resolution).is_integer():
            raise ValueError(
                f"Time per step {time_step:.2f}s"
                f"is not divisible by time resolution {time_resolution:.2f}s"
            )

        self.steps_per_time_step = int(time_step / time_resolution)
        self.time_step = time_step
        self.time_resolution = time_resolution
        self.v_cell = v_cell
        self.v_voxel = v_voxel

        self.chemistry = chemistry
        self.kinetics = kinetics or Kinetics(
            **(kinetics_kwargs or {}), dt=time_resolution, tensor_kwargs=tensor_kwargs
        )
        self.culture = culture or (
            Culture(
                **(culture_kwargs or {}),
                chemistry=chemistry,
                tensor_kwargs=tensor_kwargs,
            )
        )
        self.diffusion = diffusion or Diffusion(
            **(diffusion_kwargs or {}),
            diff_coeffs=[d.diffusivity for d in chemistry.molecules],
            N=self.culture.map_size,
            dt=time_resolution,
            tensor_kwargs=tensor_kwargs,
        )
        self.permeation = permeation or Permeation(
            **(permeation_kwargs or {}),
            perm_coeffs=[d.permeability for d in chemistry.molecules],
            v_cell=v_cell,
            v_vox=v_voxel,
            dt=time_resolution,
            tensor_kwargs=tensor_kwargs,
        )
        self.degradation = degradation or Degradation(
            half_lifes=[d.half_life for d in chemistry.molecules],
            dt=time_resolution,
            tensor_kwargs=tensor_kwargs,
        )

        if not cells:
            genomics = Genomics(**(genomics_kwargs or {}))
            proteomics = Proteomics(
                **(proteomics_kwargs or {}),
                chemistry=chemistry,
                scalar_enc_size=genomics.one_codon_map.encoding_size,
                vector_enc_size=genomics.two_codon_map.encoding_size,
                tensor_kwargs=tensor_kwargs,
            )
            self.cells = Cells(
                **(cells_kwargs or {}),
                c_max=self.culture.map_size**2,
                chemistry=chemistry,
                genomics=genomics,
                proteomics=proteomics,
                tensor_kwargs=tensor_kwargs,
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
        cell_positions = self.culture.get_cell_positions()
        if by_idx is not None:
            idx = by_idx
        if by_position is not None:
            pos = self.idxtensor(by_position)
            mask = (cell_positions == pos).all(dim=1)
            idxs = torch.argwhere(mask).flatten().tolist()
            if len(idxs) == 0:
                raise ValueError(f"Cell at {by_position} not found")
            idx = idxs[0]

        return Cell(
            world=self,
            idx=idx,
            genome=self.cells.genomes.items[idx],
            position=tuple(cell_positions[idx].tolist()),  # type: ignore
            label=self.cells.labels.items[idx],
            age=int(self.cells.ages[idx].item()),
            generation=int(self.cells.generations[idx].item()),
        )

    def spawn_cells(self, genomes: list[str]) -> torch.Tensor:
        n_new_cells = len(genomes)
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
        self.cells.update_genomes(genomes=genomes, idxs=new_idxs)

        # update other values
        self.cells.alive[new_idxs] = True
        self.cells.ages[new_idxs] = 0.0
        self.cells.generations[new_idxs] = 0
        self.cells.labels[new_idxs] = [randstr(n=12) for _ in range(n_new_cells)]

        return new_idxs

    def add_cells(self, cells: list["Cell"]) -> torch.Tensor:
        n_new_cells = len(cells)
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

        n_cells = self.cells.get_n_cells()
        self.cells.update_c(c_req=n_cells + n_new_cells)
        free_idxs = self.cells.get_free_idxs()
        new_idxs = free_idxs[:n_new_cells]

        # occupy positions
        new_pos = free_pos[:n_new_cells]
        xs = new_pos[:, 0]
        ys = new_pos[:, 1]
        self.culture.cell_map[xs, ys] = new_idxs

        # update parameters from genomes
        self.cells.update_genomes(genomes=[d.genome for d in cells], idxs=new_idxs)

        # update other parameters
        int_mols = [self.ftensor(d.int_molecules) for d in cells]
        self.cells.alive[new_idxs] = True
        self.cells.labels[new_idxs] = [d.label for d in cells]
        self.cells.molecules[new_idxs, :] = torch.stack(int_mols)
        self.cells.ages[new_idxs] = self.ftensor([d.age for d in cells])
        self.cells.generations[new_idxs] = self.itensor([d.generation for d in cells])

        return new_idxs

    def resuspend_cells(self, cell_idxs: torch.Tensor | None = None) -> None:
        if cell_idxs is None:
            cell_idxs = self.cells.get_cell_idxs()

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
            cell_idxs = self.cells.get_cell_idxs()

        if cell_idxs.numel() == 0:
            return self.idxtensor([])

        # positions are updated in place
        migrated_idxs = self.culture.undirected_cell_migration(cell_idxs=cell_idxs)

        return migrated_idxs

    def divide_cells(
        self, cell_idxs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_req = cell_idxs.numel()
        if n_req == 0:
            return self.idxtensor([]), self.idxtensor([])

        # prepare for children
        avail_idxs = self.cells.get_free_idxs()
        n_avail = avail_idxs.numel()

        if n_avail >= n_req:
            child_idxs = avail_idxs[:n_req]
        else:
            n_alive = self.cells.get_n_cells()
            self.cells.update_c(c_req=n_req + n_alive)
            avail_idxs = self.cells.get_free_idxs()
            child_idxs = avail_idxs[:n_req]

        # already places children on cell map
        parent_idxs, child_idxs = self.culture.cell_division(
            parent_idxs=cell_idxs, child_idxs=child_idxs
        )

        n_new_cells = parent_idxs.numel()
        if n_new_cells == 0:
            return self.idxtensor([]), self.idxtensor([])

        # update child kinetics parameters from genomes
        self.cells.copy_genomes(from_idxs=parent_idxs, to_idxs=child_idxs)

        # update cell parameters for children
        descendant_idxs = torch.cat([parent_idxs, child_idxs])
        self.cells.alive[child_idxs] = True
        self.cells.labels[child_idxs] = self.cells.labels[parent_idxs]
        self.cells.generations[child_idxs] = self.cells.generations[parent_idxs]
        self.cells.generations[descendant_idxs] += 1
        self.cells.ages[descendant_idxs] = 0.0

        # descendants share molecules
        self.cells.molecules[child_idxs] = self.cells.molecules[parent_idxs]
        self.cells.molecules[descendant_idxs] *= 0.5

        return parent_idxs, child_idxs

    def update_cells(self, genomes: list[str], cell_idxs: torch.Tensor) -> None:
        if len(genomes) == 0:
            return

        if len(genomes) != cell_idxs.numel():
            raise ValueError("Length of genomes of cell_idxs doesn't match")

        self.cells.update_genomes(genomes=genomes, idxs=cell_idxs)

    def kill_cells(self, cell_idxs: torch.Tensor | None = None) -> None:
        if cell_idxs is None:
            cell_idxs = self.cells.get_cell_idxs()

        if cell_idxs.numel() == 0:
            return

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
        self.cells.genomes[cell_idxs] = [""] * len(cell_idxs)
        self.cells.labels[cell_idxs] = [""] * len(cell_idxs)
        self.cells.alive[cell_idxs] = False
        self.cells.ages[cell_idxs] = 0.0
        self.cells.generations[cell_idxs] = 0
        self.cells.molecules[cell_idxs] = 0.0

        self.cells.v_max[cell_idxs] = 0.0
        self.cells.k_f[cell_idxs] = 0.0
        self.cells.k_b[cell_idxs] = 0.0
        self.cells.K_r[cell_idxs] = 0.0
        self.cells.N_f[cell_idxs] = 0
        self.cells.N_b[cell_idxs] = 0
        self.cells.N_h[cell_idxs] = 0
        self.cells.N_e[cell_idxs] = 0

    def advance_time(self) -> None:
        alive = self.cells.alive  # (c,)
        cell_molecules = self.cells.molecules  # (c, m)
        cell_idxs = self.cells.get_cell_idxs()  # (c,)
        molecule_map = self.culture.molecule_map  # (m, N, N)
        cell_positions = self.culture.get_cell_positions()  # (N², 2)
        xs = cell_positions[cell_idxs, 0]
        ys = cell_positions[cell_idxs, 1]

        for _ in range(self.steps_per_time_step):
            # degradation
            molecule_map[:] = self.degradation.degrade(x=molecule_map, m_dim=0)
            cell_molecules[:] = self.degradation.degrade(x=cell_molecules, m_dim=1)

            # diffusion
            molecule_map[:] = self.diffusion.diffuse(molecule_map)

            if not alive.any():
                continue

            # collect internal and external molecules for x
            x = torch.cat(
                [cell_molecules[alive], molecule_map[:, xs, ys].T],
                dim=1,
            )

            # permeation
            x = self.permeation.permeate(x)

            # enzymatic activity
            x = self.kinetics.get_x1(
                x0=x,
                N_e=self.cells.N_e[alive],
                N_f=self.cells.N_f[alive],
                N_b=self.cells.N_b[alive],
                N_h=self.cells.N_h[alive],
                k_f=self.cells.k_f[alive],
                k_b=self.cells.k_b[alive],
                K_r=self.cells.K_r[alive],
                v_max=self.cells.v_max[alive],
            )

            # distribute x1 to internal and external molecules
            molecule_map[:, xs, ys] = x[:, self._ext_mol_idxs].T
            cell_molecules[alive] = x[:, self._int_mol_idxs]

        # age cells
        self.cells.ages += self.time_step

    # def mutate_cells(
    #     self,
    #     p: float = 1e-6,
    #     p_indel: float = 0.4,
    #     p_del: float = 0.66,
    # ) -> None:
    #     cell_idxs = self.cells.get_free_idxs()
    #     seqs = self.cells.genomes[cell_idxs]
    #     mutated = point_mutations(seqs=seqs, p=p, p_indel=p_indel, p_del=p_del)
    #     genomes = [d[0] for d in mutated]
    #     idxs = [d[1] for d in mutated]
    #     self.update_cells(genomes=genomes, cell_idxs=cell_idxs[idxs])

    # def recombinate_cells(self, p: float = 1e-7) -> None:
    #     nghbrs = self.culture.get_cell_neighbor_pairs()
    #     genomes0 = self.cells.genomes[[d[0] for d in nghbrs]]
    #     genomes1 = self.cells.genomes[[d[1] for d in nghbrs]]
    #     pairs = [(g0, g1) for g0, g1 in zip(genomes0, genomes1)]
    #     mutated = recombinations(seq_pairs=pairs, p=p)

    #     genomes0 = [d[0] for d in mutated]
    #     genomes1 = [d[1] for d in mutated]
    #     idxs = [d[2] for d in mutated]

    #     genomes = genomes0 + genomes1
    #     cell_idxs = torch.cat([nghbrs[idxs:0], nghbrs[idxs:1]])
    #     self.update_cells(genomes=genomes, cell_idxs=cell_idxs)

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
