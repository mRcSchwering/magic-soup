from typing import Optional
import torch
from genetics import Domain, Information


class Cells:
    """State of all cells in simulation"""

    def __init__(
        self,
        molecules: list[Information],
        actions: list[Information],
        dtype=torch.float,
        device="cpu",
        max_proteins=1000,
        cell_signal_degrad=0.8,
    ):
        self.molecules = molecules
        self.actions = actions
        self.max_proteins = max_proteins
        self.cell_signal_degrad = cell_signal_degrad
        self.dtype = dtype
        self.device = device
        self.n_molecules = len(molecules)
        self.n_actions = len(actions)
        self.n_infos = self.n_molecules * 2 + self.n_actions
        self.n_cell_infos = self.n_molecules + self.n_actions

        self.cell_info_map = {d: i for i, d in enumerate(molecules + actions)}
        self.world_info_map = {
            d: i + self.n_cell_infos for i, d in enumerate(molecules)
        }

        self.genomes: list[str] = []
        self.positions: list[tuple[int, int]] = []
        self.cell_signals = torch.zeros(
            0, self.n_cell_infos, dtype=dtype, device=device
        )
        self.A = torch.zeros(0, self.n_infos, max_proteins, dtype=dtype, device=device,)
        self.B = torch.zeros(0, self.n_infos, max_proteins, dtype=dtype, device=device,)

    def add_cells(
        self,
        proteomes: list[list[dict[Domain, tuple[Optional[float], Optional[float]]]]],
        positions: list[tuple[int, int]],
    ):
        if len(positions) != len(proteomes):
            raise ValueError(
                "proteomes and positions must have the same length. "
                "They both represent proteomes, positions of the same list of cells respectively. "
                f"Now, proteomes has {len(proteomes)}, positions {len(positions)} elements."
            )
        self.positions.extend(positions)
        A, B = self.get_cell_params(cells=proteomes)
        self.A = torch.concat([self.A, A], dim=0)
        self.B = torch.concat([self.B, B], dim=0)
        self.cell_signals = self._get_cell_signals(n_cells=len(proteomes))

    def _get_cell_signals(self, n_cells: int) -> torch.Tensor:
        # TODO: get parent concentrations (for duplication events only)
        new = torch.randn(
            n_cells, self.n_cell_infos, dtype=self.dtype, device=self.device
        ).abs()
        return torch.concat([self.cell_signals, new], dim=0)

    def degrade_signals(self):
        self.cell_signals = self.cell_signals * self.cell_signal_degrad

    def get_cell_params(
        self, cells: list[list[dict[Domain, tuple[Optional[float], Optional[float]]]]]
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
        # TODO: device, dtype
        A = torch.zeros(n_cells, self.n_infos, self.max_proteins)
        B = torch.zeros(n_cells, self.n_infos, self.max_proteins)
        for cell_i, cell in enumerate(cells):
            for prot_i, protein in enumerate(cell):
                for dom, (a, b) in protein.items():
                    # TODO: check whether it needs world concentration instead
                    if dom.info:
                        idx = self.cell_info_map[dom.info]
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

