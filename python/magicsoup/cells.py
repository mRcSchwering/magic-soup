import logging
import math
from pathlib import Path

import torch

from magicsoup import rs
from magicsoup.chemistry import Chemistry
from magicsoup.genomics import Genomics
from magicsoup.proteomics import Proteomics
from magicsoup.util import Array, TensorClass

_log = logging.getLogger(__name__)


class Cells(TensorClass):

    def __init__(
        self,
        chemistry: Chemistry,
        genomics: Genomics,
        proteomics: Proteomics,
        dim_scaling: float = 0.2,
        c_max: int = 100_000,
        p_max: int = 1_000,
        device: str = "cpu",
        itype: torch.dtype = torch.int8,
        ftype: torch.dtype = torch.float32,
    ):
        super().__init__(device=device, itype=itype, ftype=ftype)

        self.chemistry = chemistry
        self.genomics = genomics
        self.proteomics = proteomics
        self.dim_scaling = dim_scaling
        self.c_max = c_max
        self.p_max = p_max

        molecules = chemistry.molecules
        n_molecules = len(molecules)
        self._int_mol_idxs = list(range(n_molecules))
        self._ext_mol_idxs = list(range(n_molecules, n_molecules * 2))
        self._mol_degrad = [self._get_degrade(m.half_life) for m in molecules]
        self._permeation = [
            self._get_permeate(mol_perm_rate=m.permeability) for m in molecules
        ]

        self.c = 0  # dim 0
        self.p = 0  # dim 1
        self.m = 2 * n_molecules  # dim2

        # cellular parameters
        self.genomes: Array[str] = Array([])  # cell genomes
        self.labels: Array[str] = Array([])  # cell labels
        self.alive = self.izeros(0).bool()  # which ones are alive
        self.positions = self.idxzeros(0, 2)  # x,y positions on map
        self.ages = self.idxzeros(0)  # time since last division
        self.generations = self.idxzeros(0)  # number of divisions
        self.molecules = self.fzeros(0, n_molecules)  # intracellular concentrations

        # kinetics parameters
        self.k_f = self.fzeros(0, 0)  # forward reaction rates
        self.k_b = self.fzeros(0, 0)  # backward reaction rates
        self.K_r = self.fzeros(0, 0, self.m)  # effector affinities
        self.v_max = self.fzeros(0, 0)  # maximum reaction rates
        self.N = self.izeros(0, 0, self.m)  # stoichiometric numbers
        self.N_f = self.izeros(0, 0, self.m)  # forward stoichiometric coefficients
        self.N_b = self.izeros(0, 0, self.m)  # backward stoichiometric coefficients
        self.N_h = self.izeros(0, 0, self.m)  # Hill coefficients

        # setup rust class
        self._setup_rs()

    def get_alive_cells(self) -> int:
        return int(self.alive.sum().item())

    def get_available_idxs(self) -> torch.Tensor:
        return (~self.alive).nonzero(as_tuple=True)[0]

    def update_genomes(self, genomes: list[str], idxs: list[int]) -> None:
        v_max_, k_m_, K_r_, N_f_, N_b_, N_h_ = self.rs.translate_genomes(
            genomes=genomes
        )
        v_max = self.ftensor(v_max_)
        k_m = self.ftensor(k_m_)
        K_r = self.ftensor(K_r_)
        N_f = self.itensor(N_f_)
        N_b = self.itensor(N_b_)
        N_h = self.itensor(N_h_)

        # new proteomes may have different p
        p_req = v_max.shape[1]
        self.update_p(p_req=p_req)

        # p have could been adjusted upwards or p_req was already < p
        if self.p > p_req:
            p_diff = self.p - p_req
            v_max = self._expand_p(v_max, by_n=p_diff)
            k_m = self._expand_p(k_m, by_n=p_diff)
            K_r = self._expand_p(K_r, by_n=p_diff)
            N_f = self._expand_p(N_f, by_n=p_diff)
            N_b = self._expand_p(N_b, by_n=p_diff)
            N_h = self._expand_p(N_h, by_n=p_diff)

        # p_req could have been > p_max
        if self.p < p_req:
            v_max = v_max[:, : self.p]
            k_m = k_m[:, : self.p]
            K_r = K_r[:, : self.p]
            N_f = N_f[:, : self.p]
            N_b = N_b[:, : self.p]
            N_h = N_h[:, : self.p]
            _log.warning("Proteome size %d has to be reduced to %d", p_req, self.p_max)

        # derive final tensors
        N = self.proteomics.derive_stoichiometry(N_f=N_f, N_b=N_b)
        k_f, k_b = self.proteomics.derive_rates(N=N, k_m=k_m)

        # update kinetics parameters
        self.N[idxs] = N
        self.N_f[idxs] = N_f
        self.N_b[idxs] = N_b
        self.N_h[idxs] = N_h
        self.k_f[idxs] = k_f
        self.k_b[idxs] = k_b
        self.K_r[idxs] = K_r
        self.v_max[idxs] = v_max

        # update cellular parameters
        self.genomes[idxs] = genomes

    def update_c(self, c_req: int) -> None:
        mar = math.ceil(c_req * self.dim_scaling)
        to_c = min(c_req + mar, self.c_max)
        if to_c < self.c:
            self._decrease_c(by_n=self.c - to_c)
        if c_req > self.c:
            self._increase_c(by_n=to_c - self.c)

    def update_p(self, p_req: int) -> None:
        mar = math.ceil(p_req * self.dim_scaling)
        to_p = min(p_req + mar, self.p_max)
        if to_p < self.p:
            self._decrease_p(by_n=self.p - to_p)
        if p_req > self.p:
            self._increase_p(by_n=to_p - self.p)

    def save_state(self, statedir: Path) -> None:
        name = type(self).__name__
        torch.save(self.alive, statedir / f"{name}.alive.pt")
        torch.save(self.ages, statedir / f"{name}.ages.pt")
        torch.save(self.positions, statedir / f"{name}.positions.pt")
        torch.save(self.generations, statedir / f"{name}.generations.pt")
        torch.save(self.molecules, statedir / f"{name}.molecules.pt")
        torch.save(self.k_f, statedir / f"{name}.k_f.pt")
        torch.save(self.k_b, statedir / f"{name}.k_b.pt")
        torch.save(self.K_r, statedir / f"{name}.K_r.pt")
        torch.save(self.v_max, statedir / f"{name}.v_max.pt")
        torch.save(self.N, statedir / f"{name}.N.pt")
        torch.save(self.N_b, statedir / f"{name}.N_b.pt")
        torch.save(self.N_f, statedir / f"{name}.N_f.pt")
        torch.save(self.N_h, statedir / f"{name}.N_h.pt")

        lines: list[str] = [
            f">{i} {l}\n{g}"
            for i, (g, l) in enumerate(zip(self.genomes.items, self.labels.items))
        ]

        with open(statedir / f"{name}.genomes.fasta", "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    def load_state(self, statedir: Path) -> None:
        name = type(self).__name__
        self.alive = torch.load(
            statedir / f"{name}.alive.pt",
            map_location=self.device,
            dtype=torch.bool,
        )
        self.ages = torch.load(
            statedir / f"{name}.ages.pt",
            map_location=self.device,
            dtype=torch.int32,
        )
        self.positions = torch.load(
            statedir / f"{name}.positions.pt",
            map_location=self.device,
            dtype=torch.int32,
        )
        self.generations = torch.load(
            statedir / f"{name}generations.pt",
            map_location=self.device,
            dtype=torch.int32,
        )
        self.molecules = torch.load(
            statedir / f"{name}.molecules.pt",
            map_location=self.device,
            dtype=self.ftype,
        )
        self.k_f = torch.load(
            statedir / f"{name}.k_f.pt",
            map_location=self.device,
            dtype=self.ftype,
        )
        self.k_b = torch.load(
            statedir / f"{name}.k_b.pt",
            map_location=self.device,
            dtype=self.ftype,
        )
        self.K_r = torch.load(
            statedir / f"{name}.K_r.pt",
            map_location=self.device,
            dtype=self.ftype,
        )
        self.v_max = torch.load(
            statedir / f"{name}.v_max.pt",
            map_location=self.device,
            dtype=self.ftype,
        )
        self.N = torch.load(
            statedir / f"{name}.N.pt",
            map_location=self.device,
            dtype=self.itype,
        )
        self.N_f = torch.load(
            statedir / f"{name}.N_f.pt",
            map_location=self.device,
            dtype=self.itype,
        )
        self.N_b = torch.load(
            statedir / f"{name}.N_b.pt",
            map_location=self.device,
            dtype=self.itype,
        )
        self.N_h = torch.load(
            statedir / f"{name}.N_h.pt",
            map_location=self.device,
            dtype=self.itype,
        )

        with open(statedir / f"{name}.genomes.fasta", encoding="utf-8") as fh:
            text: str = fh.read()
            entries = [d.strip() for d in text.split(">") if len(d.strip()) > 0]

        self.labels = Array([])
        self.genomes = Array([])
        for entry in entries:
            parts = entry.split("\n")
            descr = parts[0]
            seq = "" if len(parts) < 2 else parts[1]
            names = descr.split()
            label = names[1].strip() if len(names) > 1 else ""
            self.genomes.items.append(seq)
            self.labels.items.append(label)

    def _setup_rs(self) -> None:
        self.rs = rs.Cells(genomics=self.genomics.rs, proteomics=self.proteomics.rs)

    def _increase_c(self, by_n: int) -> None:
        # cell paramaters
        self.genomes.items.extend([""] * by_n)
        self.labels.items.extend([""] * by_n)
        self.alive = self._expand_c(t=self.alive, by_n=by_n)
        self.ages = self._expand_c(t=self.ages, by_n=by_n)
        self.positions = self._expand_c(t=self.positions, by_n=by_n)
        self.generations = self._expand_c(t=self.generations, by_n=by_n)
        self.molecules = self._expand_c(t=self.molecules, by_n=by_n)

        # kinetics parameters
        self.v_max = self._expand_c(t=self.v_max, by_n=by_n)
        self.k_f = self._expand_c(t=self.k_f, by_n=by_n)
        self.k_b = self._expand_c(t=self.k_b, by_n=by_n)
        self.K_r = self._expand_c(t=self.K_r, by_n=by_n)
        self.N_f = self._expand_c(t=self.N_f, by_n=by_n)
        self.N_b = self._expand_c(t=self.N_b, by_n=by_n)
        self.N_h = self._expand_c(t=self.N_h, by_n=by_n)
        self.N = self._expand_c(t=self.N, by_n=by_n)

        # set new c
        self.c = len(self.genomes)

    def _decrease_c(self, by_n: int) -> None:
        avail = self.get_available_idxs()
        n_avail = avail.numel()
        if n_avail < by_n:
            raise ValueError(f"Cannot decrease c by {by_n}, only {n_avail} available")

        avail = avail[:by_n]
        keep = torch.ones_like(self.alive, dtype=torch.bool)
        keep[avail] = False

        # cell parameters
        self.genomes = Array(self.genomes[keep])
        self.labels = Array(self.labels[keep])
        self.alive = self.alive[keep]
        self.ages = self.ages[keep]
        self.positions = self.positions[keep]
        self.generations = self.generations[keep]
        self.molecules = self.molecules[keep]

        # kinetics parameters
        self.v_max = self.v_max[keep]
        self.k_f = self.k_f[keep]
        self.k_b = self.k_b[keep]
        self.K_r = self.K_r[keep]
        self.N_f = self.N_f[keep]
        self.N_b = self.N_b[keep]
        self.N_h = self.N_h[keep]
        self.N = self.N[keep]

        # set new c
        self.c = len(self.genomes)

    def _increase_p(self, by_n: int) -> None:
        # kinetics parameters
        self.v_max = self._expand_p(t=self.v_max, by_n=by_n)
        self.k_f = self._expand_p(t=self.k_f, by_n=by_n)
        self.k_b = self._expand_p(t=self.k_b, by_n=by_n)
        self.K_r = self._expand_p(t=self.K_r, by_n=by_n)
        self.N_f = self._expand_p(t=self.N_f, by_n=by_n)
        self.N_b = self._expand_p(t=self.N_b, by_n=by_n)
        self.N_h = self._expand_p(t=self.N_h, by_n=by_n)
        self.N = self._expand_p(t=self.N, by_n=by_n)

        # set new p
        self.p = self.v_max.shape[1]

    def _decrease_p(self, by_n: int) -> None:
        # kinetics parameters
        self.v_max = self.v_max[:, :-by_n]
        self.k_f = self.k_f[:, :-by_n]
        self.k_b = self.k_b[:, :-by_n]
        self.K_r = self.K_r[:, :-by_n]
        self.N_f = self.N_f[:, :-by_n]
        self.N_b = self.N_b[:, :-by_n]
        self.N_h = self.N_h[:, :-by_n]
        self.N = self.N[:, :-by_n]

        # set new p
        self.p = self.v_max.shape[1]

    def _get_permeate(self, mol_perm_rate: float) -> float:
        if mol_perm_rate < 0.0:
            mol_perm_rate = -mol_perm_rate

        mol_perm_rate = min(mol_perm_rate, 1.0)

        if mol_perm_rate == 0.0:
            return 0.0

        d = 1 / mol_perm_rate
        return 1 / (d + 1)

    def _get_degrade(self, half_life: float) -> float:
        return math.exp(-math.log(2) / half_life)

    def _expand_c(self, t: torch.Tensor, by_n: int) -> torch.Tensor:
        size = t.size()
        zeros = torch.zeros(by_n, *size[1:], device=t.device, dtype=t.dtype)
        return torch.cat([t, zeros], dim=0)

    def _expand_p(self, t: torch.Tensor, by_n: int) -> torch.Tensor:
        size = t.size()
        zeros = torch.zeros(size[0], by_n, *size[2:], device=t.device, dtype=t.dtype)
        return torch.cat([t, zeros], dim=1)
