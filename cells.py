import torch
from util import trunc
from genetics import Information, Protein


# TODO: move A, B, X, Z stuff to Incubator class


class Cells:
    """State of all cells in simulation"""

    def __init__(
        self,
        molecules: list[Information],
        actions: list[Information],
        dtype=torch.float,
        device="cpu",
        n_max_proteins=1000,
        mol_degrad=0.9,
        trunc_n_decs=4,
    ):
        self.n_max_proteins = n_max_proteins
        self.mol_degrad = mol_degrad
        self.torch_kwargs = {"dtype": dtype, "device": device}
        self.trunc_n_decs = trunc_n_decs

        self.molecules = molecules
        self.actions = actions
        self.n_molecules = len(molecules)
        self.n_actions = len(actions)
        self.n_infos = self.n_molecules * 2 + self.n_actions

        # internal molecules, internal actions, external molecules
        self.in_mol_pad = 0
        self.in_act_pad = self.in_mol_pad + self.n_molecules
        self.ex_mol_pad = self.in_act_pad + self.n_actions

        self.genomes: list[str] = []
        self.positions: list[tuple[int, int]] = []
        self.molecule_map = torch.zeros(0, self.n_molecules, **self.torch_kwargs)
        self.action_map = torch.zeros(0, self.n_actions, **self.torch_kwargs)

        self.A = self._get_A(n_cells=0)
        self.B = self._get_B(n_cells=0)
        self.Z = self._get_Z(n_cells=0)

    def add_cells(
        self,
        genomes: list[str],
        proteomes: list[list[Protein]],
        positions: list[tuple[int, int]],
    ):
        if not len(positions) == len(proteomes) == len(genomes):
            raise ValueError(
                "proteomes, positions, genomes must have the same length. "
                "They both represent proteomes, positions of the same list of cells respectively. "
                f"Now, proteomes has {len(proteomes)}, positions {len(positions)}, genomes {len(genomes)} elements."
            )

        n = len(positions)
        self.genomes.extend(genomes)
        self.positions.extend(positions)
        A, B, Z = self.get_cell_params(proteomes=proteomes)
        self.A = torch.concat([self.A, A], dim=0)
        self.B = torch.concat([self.B, B], dim=0)
        self.Z = torch.concat([self.Z, Z], dim=0)

        # TODO: get parent actions, molecules (for duplication events only)
        #       replication action should be set to 0 first
        molecule_map = torch.zeros(n, self.n_molecules, **self.torch_kwargs)
        action_map = torch.zeros(n, self.n_actions, **self.torch_kwargs)
        self.molecule_map = torch.concat([self.molecule_map, molecule_map], dim=0)
        self.action_map = torch.concat([self.action_map, action_map], dim=0)

    def degrade_molecules(self):
        self.molecule_map = trunc(
            tens=self.molecule_map * self.mol_degrad, n_decs=self.trunc_n_decs
        )

    def get_cell_params(
        self, proteomes: list[list[Protein]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate tensors A, B, Z from cell proteomes

        Returns tuple (A, B, Z):

        - `A` parameters defining how incomming signals map to proteins, `(c, s, p)`
        - `B` parameters defining how much proteins can produce output signals, `(c, s, p)`
        - `Z` binary matrix which protein can produce output at all, `(p,)`
        
        where `s` is the number of signals, `p` is the number of proteins,
        and `c` is the number of cells.
        """
        n = len(proteomes)
        A = self._get_A(n_cells=n)
        B = self._get_B(n_cells=n)
        Z = self._get_Z(n_cells=n)

        for cell_i, cell in enumerate(proteomes):
            for prot_i, protein in enumerate(cell):
                # TODO: > 0 energy proteins are only relevant if transporer in it
                Z[cell_i, prot_i] = float(protein.energy <= 0)
                for dom, w in protein.domains.items():
                    if dom.info is not None:
                        if dom.info.is_molecule:
                            idx = self.molecules.index(dom.info)
                            if protein.is_transmembrane and dom.is_incomming:
                                offset = self.ex_mol_pad
                            else:
                                offset = self.in_mol_pad
                        else:
                            idx = self.actions.index(dom.info)
                            offset = self.in_act_pad
                        if dom.is_incomming:
                            A[cell_i, idx + offset, prot_i] = w
                        else:
                            B[cell_i, idx + offset, prot_i] = w
        return (A, B, Z)

    def simulate_protein_work(
        self, X: torch.Tensor, A: torch.Tensor, B: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate new information (molecules/actions) created by proteins after 1 timestep.
        Returns neew informations in shape `(c, s)` with `s` signals(=information), `p` proteins, `c` cells.

        - `X` initial incomming information, `(c, s)`
        - `A` parameters defining how incomming information maps to proteins, `(c, s, p)`
        - `B` parameters defining how much proteins can produce output information, `(c, s, p)`
        - `Z` binary matrix which protein can produce output at all, `(c, p)`

        Proteins' output is defined by:
        
            `(1 - exp(-(X * A) ** 3)) * (B * Z)`

        In `X (c, s)` signals are ordered in a specific way to reduce computation.
        First, are all molecule concentrations in order. Second, all action intensities in order.
        Both of these represent conditions within the cell. Third, are all molecule concentrations
        outside the cell. So, with 3 molecules and 2 actions, cytoplasmic molecules are in idxs
        0 to 2, cell actions in 3 to 4, extracellular molecules in 5 to 7. These signals `s` are
        ordered the same way in `A`, `B` and the return tensor.
        
        # TODO: implications of this calculation
        """
        # protein activity, matrix (c x p)
        X_1 = torch.einsum("ij,ijk->ik", X, A)
        X_2 = 1 - torch.exp(-(X_1 ** 3))

        # protein output potentials, matrix (c x s)
        B_1 = torch.einsum("ijk,ik->ijk", B, Z)

        # protein output, matrix (c x s)
        X_3 = torch.einsum("ij,ikj->ik", X_2, B_1)

        return trunc(tens=X_3, n_decs=self.trunc_n_decs)

    def get_signals(self, world_mol_map: torch.Tensor) -> torch.Tensor:
        X = self._get_X(n_cells=len(self))
        for cell_i, (x, y) in enumerate(self.positions):
            for mol_i in range(len(self.molecules)):
                X[cell_i, mol_i + self.in_mol_pad] = self.molecule_map[cell_i, mol_i]
                X[cell_i, mol_i + self.ex_mol_pad] = world_mol_map[mol_i, 0, x, y]
            for act_i in range(len(self.actions)):
                X[cell_i, act_i + self.in_act_pad] = self.action_map[cell_i, act_i]
        return X

    def update_signal_maps(self, X: torch.Tensor, world_mol_map: torch.Tensor):
        for cell_i, (x, y) in enumerate(self.positions):
            for mol_i in range(len(self.molecules)):
                self.molecule_map[cell_i, mol_i] = X[cell_i, mol_i + self.in_mol_pad]
                world_mol_map[mol_i, 0, x, y] = X[cell_i, mol_i + self.ex_mol_pad]
            for act_i in range(len(self.actions)):
                self.action_map[cell_i, act_i] = X[cell_i, act_i + self.in_act_pad]

    def _get_A(self, n_cells: int) -> torch.Tensor:
        return torch.zeros(
            n_cells, self.n_infos, self.n_max_proteins, **self.torch_kwargs
        )

    def _get_B(self, n_cells: int) -> torch.Tensor:
        return torch.zeros(
            n_cells, self.n_infos, self.n_max_proteins, **self.torch_kwargs
        )

    def _get_X(self, n_cells: int) -> torch.Tensor:
        return torch.zeros(n_cells, self.n_infos, **self.torch_kwargs)

    def _get_Z(self, n_cells: int) -> torch.Tensor:
        return torch.zeros(n_cells, self.n_max_proteins, **self.torch_kwargs)

    def __len__(self) -> int:
        return len(self.genomes)

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(n_molecules=%r,n_actions=%r)" % (
            clsname,
            self.n_molecules,
            self.n_actions,
        )

