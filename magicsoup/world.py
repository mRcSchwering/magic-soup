from typing import Optional
import random
import torch
from .proteins import Molecule, Protein
from .util import trunc, randstr, GAS_CONSTANT


# TODO: fit diffusion rate to natural diffusion rate of small molecules in cell
# TODO: summary()


class Cell:
    """
    Container for information about a cell
    """

    def __init__(self, genome: str, name: Optional[str] = None):
        self.genome = genome
        self.name = name or randstr()
        self.position: Optional[tuple[int, int]] = None
        self.n_survived_steps: Optional[int] = None
        self.molecules: Optional[torch.Tensor] = None


class World:
    """
    World definition with map that stores molecule/signal concentrations and cell
    positions. It also defines how all proteins of all cells integrate signals
    of their respective environments.

    - `molecules` list of all possible molecules that can be encoutered
    - `actions` list of all possible actions that can be encountered
    - `map_size` number of pixel in both directions
    - `mol_degrad` by how much molecules should be degraded each step
    - `mol_diff_rate` how much molecules diffuse at each step
    - `mol_map_init` how to initialize molecule maps (`randn` or `zeros`)
    - `n_max_proteins` how many proteins any single cell is expected to have at maximum at any point during the simulation
    - `trunc_n_decs` by how many decimals to truncate values after each computation
    - `dtype` pytorch dtype to use for tensors (see pytorch docs)
    - `device` pytorch device to use for tensors (e.g. `cpu` or `gpu`, see pytorch docs)

    The world map (for cells and molecules) is a square map with `map_size` pixels in both
    directions. But it is "circular" in that when you move over the border to the left, you re-appear
    on the right and vice versa (same for up and down). Each cell occupies one pixel. The cell
    experiences molecule concentrations on that pixel as external molecule concentrations. Additionally,
    it has its internal molecule concentrations.

    It is faster to setup a large tensor (filled mostly with `0.0`) than to readjust the size of the tensor
    during every round (according to how many proteins the largest cell has at the moment). Better set
    `n_max_proteins` a bit higher than expected. The simulation would raise an exception and stop
    if a cell at some point would have more proteins than `n_max_proteins`.

    Reducing `trunc_n_decs` has 2 effects. It speeds up calculations but also makes them less accurate.
    Another side effect is that certain values become zero more quickly. This could be a desired effect.
    E.g. the degradation of molecules would asymptotically approach 0. With `trunc_n_decs=4` any concentration
    at `0.0001` will become `0.0000` after the next time step.

    For `dtype` and `device` see pytorch's documentation. You can speed up computation by moving calculations
    to a GPU. If available you can use `dtype=torch.float16` which can speed up computations further.
    """

    def __init__(
        self,
        molecules: list[Molecule],
        map_size=128,
        mol_degrad=0.9,
        mol_diff_rate=1.0,
        mol_map_init="randn",
        abs_temp=310.0,
        n_max_proteins=1000,
        trunc_n_decs=4,
        device="cpu",
        dtype=torch.float,
    ):
        self.map_size = map_size
        self.abs_temp = abs_temp
        self.rt = abs_temp * GAS_CONSTANT
        self.n_max_proteins = n_max_proteins
        self.mol_degrad = mol_degrad
        self.mol_diff_rate = mol_diff_rate
        self.trunc_n_decs = trunc_n_decs
        self.torch_kwargs = {"dtype": dtype, "device": device}

        self.molecules = molecules
        self.n_molecules = len(molecules)
        self.n_signals = 2 * self.n_molecules

        # ordering of signals:
        # internal molecules, external molecules
        self.in_mol_pad = 0
        self.ex_mol_pad = self.n_molecules
        self.mol_pad = {True: self.in_mol_pad, False: self.ex_mol_pad}

        self.molecule_map = self._get_molecule_map(mol_map_init=mol_map_init)
        self.cell_map = torch.zeros(map_size, map_size, device=device, dtype=torch.bool)
        self.conv113 = self._get_conv(mol_diff_rate=mol_diff_rate)

        self.cell_molecules = self._tensor(0, self.n_molecules)

        self.cells: list[Cell] = []
        self.cell_survival = self._tensor(0)
        self.cell_positions: list[tuple[int, int]] = []

    def get_cell(
        self, by_idx: Optional[int] = None, by_name: Optional[str] = None
    ) -> Cell:
        """Get a cell with all its information by name or index"""
        if by_idx is not None:
            idx = by_idx
        if by_name is not None:
            cell_names = [d.name for d in self.cells]
            idx = cell_names.index(by_name)
            if idx < 0:
                raise ValueError(f"Cell {by_name} not found")

        cell = self.cells[idx]
        cell.position = self.cell_positions[idx]
        cell.n_survived_steps = int(self.cell_survival[idx].item())
        cell.molecules = self.cell_molecules[idx]
        return cell

    def add_cells(self, genomes: list[str]):
        """
        Randomly place cells on cell map and fill them with the molecules that
        were present at their respective pixels.
        """
        n_cells = len(genomes)
        pxls = torch.argwhere(~self.cell_map)

        try:
            idxs = random.sample(range(len(pxls)), k=n_cells)
        except ValueError as err:
            raise ValueError(
                f"Wanted to add {n_cells} cells but only {len(pxls)} pixels left on map"
            ) from err

        self.cell_positions.extend(tuple(d.tolist()) for d in pxls[idxs])  # type: ignore
        self.cells.extend(Cell(genome=d) for d in genomes)

        # TODO: get parent actions, molecules (for duplication events only)
        #       replication action should be set to 0 first

        # TODO: also doublecheck this
        xs = [d[0] for d in pxls]
        ys = [d[1] for d in pxls]
        cell_molecules = self.molecule_map[:, xs, ys].T

        cell_survival = self._tensor(n_cells)
        self.cell_molecules = torch.concat([self.cell_molecules, cell_molecules], dim=0)
        self.cell_survival = torch.concat([self.cell_survival, cell_survival], dim=0)

    @torch.no_grad()
    def diffuse_molecules(self):
        """Let molecules in world map diffuse by 1 time step"""
        before = self.molecule_map.unsqueeze(1)
        after = trunc(tens=self.conv113(before), n_decs=self.trunc_n_decs)
        self.molecule_map = torch.squeeze(after, 1)

    def degrade_molecules(self):
        """Truncate molecules in world map and cells by 1 time step"""
        self.molecule_map = trunc(
            tens=self.molecule_map * self.mol_degrad, n_decs=self.trunc_n_decs
        )
        self.cell_molecules = trunc(
            tens=self.cell_molecules * self.mol_degrad, n_decs=self.trunc_n_decs
        )

    def increment_cell_survival(self):
        """Increment number of current cells' time steps by 1"""
        self.cell_survival = self.cell_survival + 1

    def get_signals(self) -> torch.Tensor:
        """
        Create signals tensor for all cells from all sources of signals
        """
        X = self._tensor(len(self.cells), self.n_signals)
        for cell_i, (x, y) in enumerate(self.cell_positions):
            for mol_i in range(len(self.molecules)):
                X[cell_i, mol_i + self.in_mol_pad] = self.cell_molecules[cell_i, mol_i]
                X[cell_i, mol_i + self.ex_mol_pad] = self.molecule_map[mol_i, x, y]
        return X

    def update_signals(self, X: torch.Tensor):
        """
        Update all sources of signals for all cells with signals tensor `X`

        - `X` new signals tensor of shape `(c, s)` (`c` cells, `s` signals)

        This method relies on the ordering of cells and signals to reduce computation.
        If cells or signals are somehow shuffled this method will silently return
        wrong results.
        """
        for cell_i, (x, y) in enumerate(self.cell_positions):
            for mol_i in range(len(self.molecules)):
                self.cell_molecules[cell_i, mol_i] = X[cell_i, mol_i + self.in_mol_pad]
                self.molecule_map[mol_i, x, y] = X[cell_i, mol_i + self.ex_mol_pad]

    def get_cell_params(
        self, proteomes: list[list[Protein]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate cell-based parameter tensors from proteomes.
        Returns `(Km, Vmax, Ke, N, A)`:

        - `X` Signal/molecule concentrations
        - `Km` Domain affinities of all proteins for each signal
        - `Vmax` Maximum velocities of all proteins
        - `Ke` Equilibrium constants of all proteins. If the current reaction quotient of a protein
        exceeds its equilibrium constant, it will work in the other direction
        - `N` Reaction stoichiometry of all proteins for each signal. Numbers < 0.0 indicate this
        amount of this molecule is being used up by the protein. Numbers > 0.0 indicate this amount of this
        molecule is being created by the protein. 0.0 means this molecule is not part of the reaction of
        this protein.
        - `A` allosteric control of all proteins for each signal.
        1.0 means this molecule acts as an activating effector on this protein. -1.0 means this molecule
        acts as an inhibiting effector on this protein. 0.0 means this molecule does not allosterically
        effect the protein.
        """
        # TODO: create on CPU then send to GPU?
        n_cells = len(proteomes)
        Km = self._tensor(n_cells, self.n_max_proteins, self.n_signals)
        Vmax = self._tensor(n_cells, self.n_max_proteins)
        E = self._tensor(n_cells, self.n_max_proteins)
        N = self._tensor(n_cells, self.n_max_proteins, self.n_signals)
        A = self._tensor(n_cells, self.n_max_proteins, self.n_signals)

        for cell_i, cell in enumerate(proteomes):
            for prot_i, protein in enumerate(cell):
                energy = 0.0
                km: list[list[float]] = [[] for _ in range(self.n_signals)]
                vmax: list[float] = []
                a: list[int] = [0 for _ in range(self.n_signals)]
                n: list[int] = [0 for _ in range(self.n_signals)]

                for dom in protein.domains:
                    if dom.is_allosteric:
                        mol = dom.substrates[0]
                        mol_i = self._molidx(mol)
                        km[mol_i].append(dom.affinity)
                        a[mol_i] += -1 if dom.is_inhibiting else 1

                    if dom.is_transporter:
                        vmax.append(dom.velocity)

                        if dom.orientation:
                            sub = dom.substrates[0]
                            prod = dom.products[0]
                        else:
                            sub = dom.products[0]
                            prod = dom.substrates[0]

                        mol_i = self._molidx(sub)
                        km[mol_i].append(dom.affinity)
                        n[mol_i] -= 1

                        mol_i = self._molidx(prod)
                        km[mol_i].append(1 / dom.affinity)
                        n[mol_i] += 1

                    if dom.is_catalytic:
                        vmax.append(dom.velocity)

                        if dom.orientation:
                            energy += dom.energy
                            subs = dom.substrates
                            prods = dom.products
                        else:
                            energy -= dom.energy
                            subs = dom.products
                            prods = dom.substrates

                        for mol in subs:
                            mol_i = self._molidx(mol)
                            km[mol_i].append(dom.affinity)
                            n[mol_i] -= 1

                        for mol in prods:
                            mol_i = self._molidx(mol)
                            km[mol_i].append(1 / dom.affinity)
                            n[mol_i] += 1

                E[cell_i, prot_i] = energy

                if len(vmax) > 0:
                    Vmax[cell_i, prot_i] = sum(vmax) / len(vmax)

                for mol_i in range(self.n_signals):
                    A[cell_i, prot_i, mol_i] = float(a[mol_i])
                    N[cell_i, prot_i, mol_i] = float(n[mol_i])

                    if len(km[mol_i]) > 0:
                        Km[cell_i, prot_i, mol_i] = sum(km[mol_i]) / len(km[mol_i])

        Ke = torch.exp(-E / self.rt)
        A = A.clamp(-1.0, 1.0)

        return (Km, Vmax, Ke, N, A)

    def integrate_signals(
        self,
        X: torch.Tensor,
        Km: torch.Tensor,
        Vmax: torch.Tensor,
        Ke: torch.Tensor,
        N: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate new molecules constructed or deconstructed by all proteins of all cells
        after 1 timestep. Returns the change in signals in shape `(c, s)`.
        There are `c` cells, `p` proteins, `s` signals/molecules.

        - `X` Signal/molecule concentrations (c, s). Must all be >= 0.0.
        - `Km` Domain affinities of all proteins for each signal (c, p, s). Must all be >= 0.0.
        - `Vmax` Maximum velocities of all proteins (c, p). Must all be >= 0.0.
        - `Ke` Equilibrium constants of all proteins (c, p). If the current reaction quotient of a protein
        exceeds its equilibrium constant, it will work in the other direction. Must all be >= 0.0.
        - `N` Reaction stoichiometry of all proteins for each signal (c, p, s). Numbers < 0.0 indicate this
        amount of this molecule is being used up by the protein. Numbers > 0.0 indicate this amount of this
        molecule is being created by the protein. 0.0 means this molecule is not part of the reaction of
        this protein.
        - `A` allosteric control of all proteins for each signal (c, p, s). Must all be -1.0, 0.0, or 1.0.
        1.0 means this molecule acts as an activating effector on this protein. -1.0 means this molecule
        acts as an inhibiting effector on this protein. 0.0 means this molecule does not allosterically
        effect the protein.
        
        Everything is based on Michaelis Menten kinetics where protein velocity
        depends on substrate concentration:

        ```
            v = Vmax * x / (Km + x)
        ```

        `Vmax` is the maximum velocity of the protein and `Km` the substrate affinity. Multiple
        substrates create interaction terms such as:

        ```
            v = Vmax * x1 * / (Km1 + x1) * x2 / (Km2 + x2)
        ```

        Allosteric effectors work non-competitively such that they reduce or raise `Vmax` but
        leave any `Km` unaffected. Effectors themself also use Michaelis Menten kinteics.
        Here is substrate `x` with inhibitor `i`:
        
        ```
            v = Vmax * x / (Kmx + x) * (1 - Vi)
            Vi = i / (Kmi + i)
        ```

        Activating effectors effectively make the protein dependent on that activator:

        ```
            v = Vmax * x / (Kmx + x) * Va
            Va = a / (Kma + a)
        ```

        Multiple effectors are summed up in `Vi` and `Va` and each is clamped to `[0;1]`
        before being multiplied with `Vmax`.

        ```
            v = Vmax * x / (Km + x) * Va * (1 - Vi)
            Va = Va1 + Va2 + ...
            Vi = Vi1 + Vi2 + ...
        ```

        Equilibrium constants `Ke` define whether the reaction of a protein (as defined in `N`)
        can take place. If the reaction quotient (the combined concentrations of all products
        devided by all substrates) is smaller than its `Ke` the reaction proceeds. If it is
        greater than `Ke` the reverse reaction will take place (`N * -1.0` for this protein).
        The idea is to link signal integration to energy conversion.

        ```
            -dG0 = R * T * ln(Ke)
            Ke = exp(-dG0 / R / T)
        ```

        where `dG0` is the standard Gibb's free energy of this reaction, `R` is gas constant,
        `T` is absolute temperature.

        There's currently a chance that proteins in a cell can deconstruct more
        of a molecule than available. This would lead to a negative concentration
        in the resulting signals `X`. To avoid this, there is a correction heuristic
        which will downregulate these proteins by so much, that the molecule will
        only be reduced to 0.

        Limitations:
        - all based on Michaelis-Menten kinetics, no cooperativity
        - all allosteric control is non-competitive (activating or inhibiting)
        - there are substrate-substrate interactions but no interactions among effectors
        - 1 protein can have multiple substrates and products but there is only 1 Km for each type of molecule
        - there can only be 1 effector per molecule (e.g. not 2 different allosteric controls for the same type of molecule)
        """
        # TODO: refactor:
        #       - it seems like some of the masks are unnecessary (e.g. if I know that in
        #         N there is 0.0 in certain places, do I really need a mask like sub_M?)
        #       - can I first do the nom / denom, then pow and prod afterwards?
        #       - are there some things which I basically calculate 2 times? Can I avoid it?
        #         or are there some intermediates which could be reused if I would calculate
        #         them slightly different/in different order?
        #       - does it help to use masked tensors? (pytorch.org/docs/stable/masked.html)
        #       - better names, split into a few functions?

        # substrates
        sub_M = torch.where(N < 0.0, 1.0, 0.0)  # (c, p s)
        sub_X = torch.einsum("cps,cs->cps", sub_M, X)  # (c, p, s)
        sub_N = torch.where(N < 0.0, -N, 0.0)  # (c, p, s)

        # products
        pro_M = torch.where(N > 0.0, 1.0, 0.0)  # (c, p s)
        pro_X = torch.einsum("cps,cs->cps", pro_M, X)  # (c, p, s)
        pro_N = torch.where(N > 0.0, N, 0.0)  # (c, p, s)

        # quotients
        nom = torch.pow(pro_X, pro_N).prod(2)  # (c, p)
        denom = torch.pow(sub_X, sub_N).prod(2)  # (c, p)
        Q = nom / denom  # (c, p)

        # adjust direction
        adj_N = torch.where(Q - Ke > 0.0, -1.0, 1.0)  # (c, p)
        N_adj = torch.einsum("cps,cp->cps", N, adj_N)

        # inhibitors
        inh_M = torch.where(A < 0.0, 1.0, 0.0)  # (c, p, s)
        inh_X = torch.einsum("cps,cs->cps", inh_M, X)  # (c, p, s)
        inh_V = torch.nansum(inh_X / (Km + inh_X), dim=2).clamp(0, 1)  # (c, p)

        # activators
        act_M = torch.where(A > 0.0, 1.0, 0.0)  # (c, p, s)
        act_X = torch.einsum("cps,cs->cps", act_M, X)  # (c, p, s)
        act_V = torch.nansum(act_X / (Km + act_X), dim=2).clamp(0, 1)  # (c, p)
        act_V_adj = torch.where(act_M.sum(dim=2) > 0, act_V, 1.0)  # (c, p)

        # substrates
        sub_M = torch.where(N_adj < 0.0, 1.0, 0.0)  # (c, p s)
        sub_X = torch.einsum("cps,cs->cps", sub_M, X)  # (c, p, s)
        sub_N = torch.where(N_adj < 0.0, -N_adj, 0.0)  # (c, p, s)

        # proteins
        prot_Vmax = sub_M.max(dim=2).values * Vmax  # (c, p)
        nom = torch.pow(sub_X, sub_N).prod(2)  # (c, p)
        denom = torch.pow(Km + sub_X, sub_N).prod(2)  # (c, p)
        prot_V = prot_Vmax * nom / denom * (1 - inh_V) * act_V_adj  # (c, p)

        # concentration deltas (c, s)
        Xd = torch.einsum("cps,cp->cs", N_adj, prot_V)

        X1 = X + Xd
        if torch.any(X1 < 0):
            neg_conc = torch.where(X1 < 0, 1.0, 0.0)  # (c, s)
            candidates = torch.where(N_adj < 0.0, 1.0, 0.0)  # (c, p, s)

            # which proteins need to be down-regulated (c, p)
            BX_mask = torch.einsum("cps,cs->cps", candidates, neg_conc)
            prot_M = BX_mask.max(dim=2).values

            # what are the correction factors (c,)
            correct = torch.where(neg_conc > 0.0, -X / Xd, 1.0).min(dim=1).values

            # correction for protein velocities (c, p)
            prot_V_adj = torch.einsum("cp,c->cp", prot_M, correct)
            prot_V_adj = torch.where(prot_V_adj == 0.0, 1.0, prot_V_adj)

            # new concentration deltas (c, s)
            Xd = torch.einsum("cps,cp->cs", N_adj, prot_V * prot_V_adj)

            # TODO: can I already predict this before calculated Xd the first time?
            #       maybe at the time when I know the intended protein velocities?

        # TODO: correct for X1 Q -Ke changes
        #       if the new concentration (x1) would have changed the direction
        #       of the reaction (a -> b) -> (b -> a), I would have a constant
        #       back and forth between the 2 reactions always overshooting the
        #       actual equilibrium
        #       It would be better in this case to arrive at Q = Ke, so that
        #       the reaction stops with x1
        #       could be a correction term (like X1 < 0) or better solution
        #       that can be calculated ahead of time

        return Xd

    def _get_molecule_map(self, mol_map_init: str) -> torch.Tensor:
        args = [self.n_molecules, self.map_size, self.map_size]
        if mol_map_init == "zeros":
            return self._tensor(*args)
        if mol_map_init == "randn":
            return torch.randn(*args, **self.torch_kwargs).abs()
        raise ValueError(f"Didnt recognize mol_map_init={mol_map_init}")

    def _get_conv(self, mol_diff_rate: float) -> torch.nn.Conv2d:
        if not 0.0 <= mol_diff_rate <= 1.0:
            raise ValueError(
                "Diffusion rate must be between 0 and 1. "
                f"Now it's mol_diff_rate={mol_diff_rate}"
            )

        if mol_diff_rate == 0.0:
            a = 0.0
            b = 1.0
        else:
            d = 1 / mol_diff_rate
            a = 1 / (d + 8)
            b = d * a

        # fmt: off
        kernel = torch.tensor([[[
            [a, a, a],
            [a, b, a],
            [a, a, a],
        ]]])
        # fmt: on

        conv = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
            **self.torch_kwargs,
        )
        conv.weight = torch.nn.Parameter(kernel, requires_grad=False)
        return conv

    def _tensor(self, *args) -> torch.Tensor:
        return torch.zeros(*args, **self.torch_kwargs)

    def _molidx(self, mol: Molecule) -> int:
        pad = self.mol_pad[mol.is_intracellular]
        return self.molecules.index(mol) + pad

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return (
            "%s(molecules=%r,map_size=%r,abs_temp=%r,mol_degrad=%r,mol_diff_rate=%r,n_max_proteins=%r,trunc_n_decs=%r,n_cells=%s)"
            % (
                clsname,
                self.molecules,
                self.map_size,
                self.abs_temp,
                self.mol_degrad,
                self.mol_diff_rate,
                self.n_max_proteins,
                self.trunc_n_decs,
                len(self.cells),
            )
        )
