import torch
from util import trunc
from .genetics import Protein, Signal


class Mechanistics:
    """
    Defines how all proteins of all cells integrate signals
    of their respective environments.

    - `molecules` list of all possible molecules that can be encoutered
    - `actions` list of all possible actions that can be encountered
    - `n_max_proteins` how many proteins any single cell is expected to have at maximum at any point during the simulation
    - `trunc_n_decs` by how many decimals to truncate values after each computation
    - `dtype` pytorch dtype to use for tensors (see pytorch docs)
    - `device` pytorch device to use for tensors (e.g. `cpu` or `gpu`, see pytorch docs)

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
        molecules: list[Signal],
        actions: list[Signal],
        n_max_proteins=1000,
        trunc_n_decs=4,
        dtype=torch.float,
        device="cpu",
    ):
        self.n_max_proteins = n_max_proteins
        self.trunc_n_decs = trunc_n_decs
        self.torch_kwargs = {"dtype": dtype, "device": device}

        self.molecules = molecules
        self.actions = actions
        self.n_molecules = len(molecules)
        self.n_actions = len(actions)
        self.n_signals = self.n_molecules * 2 + self.n_actions

        # ordering of signals:
        # internal molecules, internal actions, external molecules
        self.in_mol_pad = 0
        self.in_act_pad = self.in_mol_pad + self.n_molecules
        self.ex_mol_pad = self.in_act_pad + self.n_actions

    def get_cell_params(
        self, proteomes: list[list[Protein]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate tensors A, B, Z from cell proteomes

        Returns tuple (A, B, Z):

        - `A` parameters defining how incomming signals map to proteins, `(c, s, p)`
        - `B` parameters defining how much proteins can produce output signals, `(c, s, p)`
        - `Z` binary matrix defining which protein can produce output at all, `(c, p)`
        
        where `s` is the number of signals, `p` is the number of proteins,
        and `c` is the number of cells.
        """
        n_cells = len(proteomes)
        A = self._tensor(n_cells, self.n_signals, self.n_max_proteins)
        B = self._tensor(n_cells, self.n_signals, self.n_max_proteins)
        Z = self._tensor(n_cells, self.n_max_proteins)
        for cell_i, cell in enumerate(proteomes):
            for prot_i, protein in enumerate(cell):
                net_energy = 0.0
                for dom in protein.domains:
                    net_energy += dom.energy
                    if dom.signal.is_molecule:
                        idx = self.molecules.index(dom.signal)
                        if dom.is_transmembrane:
                            offset = self.ex_mol_pad
                        else:
                            offset = self.in_mol_pad
                    else:
                        idx = self.actions.index(dom.signal)
                        offset = self.in_act_pad
                    if dom.is_receptor:
                        A[cell_i, idx + offset, prot_i] = dom.weight
                    else:
                        B[cell_i, idx + offset, prot_i] = dom.weight
                if net_energy <= 0:
                    Z[cell_i, prot_i] = 1.0
        return (A, B, Z)

    def integrate_signals(
        self, X: torch.Tensor, A: torch.Tensor, B: torch.Tensor, Z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate new signals (molecules/actions) created by all proteins of all cells
        after 1 timestep. Returns the change in signals in shape `(c, s)`.

        - `X` initial signals tensor, `(c, s)`
        - `A` parameters defining how incomming signals map to proteins, `(c, s, p)`
        - `B` parameters defining how much proteins can produce output signals, `(c, s, p)`
        - `Z` binary matrix defining which protein can produce output at all, `(c, p)`

        where `s` signals, `p` proteins, `c` cells.
        Proteins' output is defined by:
        
        ```
            f(X, A, B, Z) = sig(X * A) * (B * Z)
            
            sig(x) = max(1, min(0, 1 - exp(-x ** 3)))
        ```

        In `X (c, s)` signals are ordered in a specific way to reduce computation.
        First, are all molecule concentrations in order. Second, all action intensities in order.
        Both of these represent conditions within the cell. Third, are all molecule concentrations
        outside the cell. So, with 3 molecules and 2 actions, cytoplasmic molecules are in idxs
        0 to 2, cell actions in 3 to 4, extracellular molecules in 5 to 7. These signals `s` are
        ordered the same way in `A`, `B` and the return tensor.
        """
        # protein activity, matrix (c x p)
        P_act = torch.einsum("ij,ijk->ik", X, A)
        P_act = torch.clamp(1 - torch.exp(-(P_act ** 3)), min=0, max=1)

        # protein output potentials, matrix (c x s x p)
        B_adj = torch.einsum("ijk,ik->ijk", B, Z)

        # protein output, matrix (c x s)
        X_delta = torch.einsum("ij,ikj->ik", P_act, B_adj)

        # check for possible negative concentrations
        X_1 = X + X_delta
        if torch.any(X_1 < 0):

            # which signals will be negative (c, s)
            X_mask = torch.where(X_1 < 0, 1.0, 0.0)

            # which values in B could be responsible (c, s, p)
            B_mask = torch.where(B_adj < 0.0, 1.0, 0.0)

            # which proteins need to be down-regulated (c, p)
            BX_mask = torch.einsum("ijk,ij->ijk", B_mask, X_mask)
            Z_mask = BX_mask.max(dim=1).values

            # what are the correction factors (c,)
            correct = torch.where(X_mask > 0.0, -X / X_delta, 1.0).min(dim=1).values

            # correction matrix for B (c, p)
            Z_adj = torch.einsum("ij,i->ij", Z_mask, correct)
            Z_adj = torch.where(Z_adj == 0.0, 1.0, Z_adj)

            # new protein output, matrix (c x s)
            B_adj = torch.einsum("ijk,ik->ijk", B_adj, Z_adj)
            X_delta = torch.einsum("ij,ikj->ik", P_act, B_adj)

        return trunc(tens=X_delta, n_decs=self.trunc_n_decs)

    # TODO: instead of world map: ordered list of signals (same order as cells)
    #       no need for positions then (just like cell_mol and cell_act)
    def get_signals(
        self,
        cell_positions: list[tuple[int, int]],
        cell_act_map: torch.Tensor,
        cell_mol_map: torch.Tensor,
        world_mol_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create signals tensor for all cells from all sources of signals

        - `cell_positions` ordered list of positions for each cell
        - `cell_act_map` ordered matrix of action signals internal to cells
        - `cell_mol_map` ordered matrix of molecule signals internal to cells
        - `world_mol_map` ordered matrix of molecule signals on the world map
          at the positions where respective cells currently are

        This method relies on the ordering of cells and signals to reduce computation.
        If cells or signals are somehow shuffled this method will silently return
        wrong results.
        """
        X = self._tensor(len(cell_positions), self.n_signals)
        for cell_i, (x, y) in enumerate(cell_positions):
            for mol_i in range(len(self.molecules)):
                X[cell_i, mol_i + self.in_mol_pad] = cell_mol_map[cell_i, mol_i]
                X[cell_i, mol_i + self.ex_mol_pad] = world_mol_map[mol_i, 0, x, y]
            for act_i in range(len(self.actions)):
                X[cell_i, act_i + self.in_act_pad] = cell_act_map[cell_i, act_i]
        return X

    def update_signal_maps(
        self,
        X: torch.Tensor,
        cell_positions: list[tuple[int, int]],
        cell_act_map: torch.Tensor,
        cell_mol_map: torch.Tensor,
        world_mol_map: torch.Tensor,
    ):
        """
        Update all sources of signals for all cells with signals tensor `X`

        - `X` new signals tensor of shape `(c, s)` (`c` cells, `s` signals)
        - `cell_positions` ordered list of positions for each cell
        - `cell_act_map` ordered matrix of action signals internal to cells
        - `cell_mol_map` ordered matrix of molecule signals internal to cells
        - `world_mol_map` ordered matrix of molecule signals on the world map
          at the positions where respective cells currently are

        This method relies on the ordering of cells and signals to reduce computation.
        If cells or signals are somehow shuffled this method will silently return
        wrong results.
        """
        for cell_i, (x, y) in enumerate(cell_positions):
            for mol_i in range(len(self.molecules)):
                cell_mol_map[cell_i, mol_i] = X[cell_i, mol_i + self.in_mol_pad]
                world_mol_map[mol_i, 0, x, y] = X[cell_i, mol_i + self.ex_mol_pad]
            for act_i in range(len(self.actions)):
                cell_act_map[cell_i, act_i] = X[cell_i, act_i + self.in_act_pad]

    def _tensor(self, *args) -> torch.Tensor:
        return torch.zeros(*args, **self.torch_kwargs)

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(n_molecules=%r,n_actions=%r)" % (
            clsname,
            self.n_molecules,
            self.n_actions,
        )

