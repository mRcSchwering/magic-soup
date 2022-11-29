import torch
from genetics import Information, Molecule, Protein, ReceptorDomain


class Cells:
    """State of all cells in simulation"""

    def __init__(
        self,
        molecules: list[Information],
        actions: list[Information],
        dtype=torch.float,
        device="cpu",
        n_max_proteins=1000,
        cell_mol_degrad=0.9,
    ):
        self.n_max_proteins = n_max_proteins
        self.cell_mol_degrad = cell_mol_degrad
        self.dtype = dtype
        self.device = device

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
        A, B = self.get_cell_params(cells=proteomes)
        self.A = torch.concat([self.A, A], dim=0)
        self.B = torch.concat([self.B, B], dim=0)

        # TODO: get parent concentrations (for duplication events only)
        kwargs = {"dtype": self.dtype, "device": self.device}
        molecule_map = torch.zeros(n, self.n_molecules, **kwargs)
        action_map = torch.zeros(n, self.n_actions, **kwargs)
        self.molecule_map = torch.concat([self.molecule_map, molecule_map], dim=0)
        self.action_map = torch.concat([self.action_map, action_map], dim=0)

    def degrade_molecules(self):
        self.molecule_map = self.molecule_map * self.cell_mol_degrad

    # TODO: need C, Z, X
    def get_cell_params(
        self, cells: list[list[Protein]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate matrices A and B from cell proteomes

        Returns tuple (A, B) with matrices both of shape (c, s, p) where
        s is the number of signals, p is the number of proteins, and c is
        the number of cells.

        21.11.22 getting cell params for 1000 proteomes each of genome
        size (1000, 5000) took 0.01s
        """
        n_cells = len(cells)
        kwagrs = {"dtype": self.dtype, "device": self.device}
        A = torch.zeros(n_cells, self.n_infos, self.n_max_proteins, **kwagrs)
        B = torch.zeros(n_cells, self.n_infos, self.n_max_proteins, **kwagrs)
        # TODO: where did I need B ones?!
        for cell_i, cell in enumerate(cells):
            for prot_i, protein in enumerate(cell):
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
        return (A, B)

    def simulate_protein_work(
        self, C: torch.Tensor, A: torch.Tensor, B: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate molecules/signals created/activated by proteins after 1 timestep.
        Returns additional concentrations in shape `(c, s)`.

        - `C` initial concentrations shape `(c, s)`
        - `A` parameters for A `(c, s, p)`
        - `B` parameters for B `(c, s, p)`

        Proteins' activation function is `B * (1 - exp(-(C * A) ** 3))`.
        There are s signals, p proteins, c cells. The way how signals are
        integrated by multiplying with A we basically assume signals all
        activate and/or competitively inhibit the protein's domain.

        21.11.22 simulating protein work for 1000 cells each
        of genome size (1000, 5000) took 0.00s excluding other functions.
        """
        # matrix (c x p)
        x = torch.einsum("ij,ijk->ik", C, A)
        y = 1 - torch.exp(-(x ** 3))

        # matrix (c x m)
        return torch.einsum("ij,ikj->ik", y, B)

