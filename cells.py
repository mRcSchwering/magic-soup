import torch
from genetics import Information, Molecule, Protein, ReceptorDomain


# TODO: torch sparse matrices?


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
        self.dtype = dtype
        self.device = device
        self.trunc_n_decs = trunc_n_decs

        self.molecules = molecules
        self.actions = actions
        self.n_molecules = len(molecules)
        self.n_actions = len(actions)
        self.n_infos = self.n_molecules * 2 + self.n_actions

        self.info_2_cell_mol_idx = {}
        self.info_2_cell_act_idx = {}
        self.info_2_world_mol_idx = {}
        idx = 0
        for info in molecules:
            self.info_2_cell_mol_idx[info] = idx
            idx += 1
        for info in actions:
            self.info_2_cell_act_idx[info] = idx
            idx += 1
        for info in molecules:
            self.info_2_world_mol_idx[info] = idx
            idx += 1

        self.genomes: list[str] = []
        self.positions: list[tuple[int, int]] = []

        kwargs = {"dtype": dtype, "device": device}
        self.molecule_map = torch.zeros(0, self.n_molecules, **kwargs)
        self.action_map = torch.zeros(0, self.n_actions, **kwargs)
        self.A = torch.zeros(0, self.n_infos, n_max_proteins, **kwargs)
        self.B = torch.zeros(0, self.n_infos, n_max_proteins, **kwargs)
        self.Z = torch.zeros(0, self.n_max_proteins, **kwargs)

    def add_cells(
        self, proteomes: list[list[Protein]], positions: list[tuple[int, int]],
    ):
        if len(positions) != len(proteomes):
            raise ValueError(
                "proteomes and positions must have the same length. "
                "They both represent proteomes, positions of the same list of cells respectively. "
                f"Now, proteomes has {len(proteomes)}, positions {len(positions)} elements."
            )

        n = len(positions)
        self.positions.extend(positions)
        A, B, Z = self.get_cell_params(proteomes=proteomes)
        self.A = torch.concat([self.A, A], dim=0)
        self.B = torch.concat([self.B, B], dim=0)
        self.Z = torch.concat([self.Z, Z], dim=0)

        # TODO: get parent concentrations (for duplication events only)
        kwargs = {"dtype": self.dtype, "device": self.device}
        molecule_map = torch.zeros(n, self.n_molecules, **kwargs)
        action_map = torch.zeros(n, self.n_actions, **kwargs)
        self.molecule_map = torch.concat([self.molecule_map, molecule_map], dim=0)
        self.action_map = torch.concat([self.action_map, action_map], dim=0)

    def degrade_molecules(self):
        self.molecule_map = self.truncate(self.molecule_map * self.mol_degrad)

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
        kwargs = {"dtype": self.dtype, "device": self.device}
        A = torch.zeros(n, self.n_infos, self.n_max_proteins, **kwargs)
        B = torch.zeros(n, self.n_infos, self.n_max_proteins, **kwargs)
        Z = torch.zeros(n, self.n_max_proteins, **kwargs)

        for cell_i, cell in enumerate(proteomes):
            for prot_i, protein in enumerate(cell):
                # TODO: > 0 energy proteins are only relevant if transporer in it
                Z[cell_i, prot_i] = float(protein.energy <= 0)
                for dom, (a, b) in protein.domains.items():
                    if dom.info is not None:
                        if isinstance(dom.info, Molecule):
                            if protein.is_transmembrane and isinstance(
                                dom, ReceptorDomain
                            ):
                                idx = self.info_2_world_mol_idx[dom.info]
                            else:
                                idx = self.info_2_cell_mol_idx[dom.info]
                        else:
                            idx = self.info_2_cell_act_idx[dom.info]
                        if a is not None:
                            A[cell_i, idx, prot_i] = a
                        if b is not None:
                            B[cell_i, idx, prot_i] = b
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
        
        # TODO: implications of this calculation
        """
        # protein activity, matrix (c x p)
        X_1 = torch.einsum("ij,ijk->ik", X, A)
        X_2 = 1 - torch.exp(-(X_1 ** 3))

        # protein output potentials, matrix (c x s)
        B_1 = torch.einsum("ijk,ik->ijk", B, Z)

        # protein output, matrix (c x s)
        X_3 = torch.einsum("ij,ikj->ik", X_2, B_1)

        return self.truncate(X_3)

    def truncate(self, tens: torch.Tensor) -> torch.Tensor:
        return torch.round(tens * 10 ** self.trunc_n_decs) / (10 ** self.trunc_n_decs)
