from typing import Any

from magicsoup import _lib  # type: ignore
from magicsoup.constants import ProteinSpecType

# util


def dist_1d(a: int, b: int, m: int) -> int:
    return _lib.dist_1d(a, b, m)


def reverse_complement(seq: str) -> str:
    return _lib.reverse_complement(seq)


# mutations


def point_mutations(
    seqs: list[str], p: float, p_indel: float, p_del: float
) -> list[tuple[str, int]]:
    return _lib.point_mutations(seqs, p, p_indel, p_del)


def recombinations(
    seq_pairs: list[tuple[str, str]], p: float
) -> list[tuple[str, str, int]]:
    return _lib.recombinations(seq_pairs, p)


# culture


class Culture:

    def __init__(self, size: int) -> None:
        self._cls = _lib.Culture(size)

    def move_cells(
        self, cell_idxs: list[int], positions: list[tuple[int, int]]
    ) -> tuple[list[tuple[int, int]], list[int]]:
        return self._cls.move_cells(cell_idxs, positions)

    def free_moores_nghbhd(
        self, x: int, y: int, positions: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        return self._cls.free_moores_nghbhd(x, y, positions)

    def get_neighbors(
        self, from_idxs: list[int], to_idxs: list[int], positions: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        return self._cls.get_neighbors(from_idxs, to_idxs, positions)

    def divide_cells_if_possible(
        self, cell_idxs: list[int], positions: list[tuple[int, int]], n_cells: int
    ) -> tuple[list[int], list[int], list[tuple[int, int]]]:
        return self._cls.divide_cells_if_possible(cell_idxs, positions, n_cells)


# genomics


class Genomics:

    def __init__(
        self,
        start_codons: list[str],
        stop_codons: list[str],
        domain_map: dict[str, int],
        one_codon_map: dict[str, int],
        two_codon_map: dict[str, int],
        dom_size: int,
        dom_type_size: int,
    ) -> None:
        self._cls = _lib.Genomics(
            start_codons,
            stop_codons,
            domain_map,
            one_codon_map,
            two_codon_map,
            dom_size,
            dom_type_size,
        )

    def translate_genomes(self, genomes: list[str]) -> list[list[ProteinSpecType]]:
        return self._cls.translate_genomes(genomes)

    def get_coding_regions(
        self,
        seq: str,
        min_cds_size: int,
        is_fwd: bool,
    ) -> list[tuple[int, int, bool]]:
        return self._cls.get_coding_regions(seq, min_cds_size, is_fwd)

    def extract_domains(
        self,
        genome: str,
        cdss: list[tuple[int, int, bool]],
    ) -> list[ProteinSpecType]:
        return self._cls.extract_domains(genome, cdss)


# proteomics


class Proteomics:

    def __init__(
        self,
        km_map: dict[int, float],
        vmax_map: dict[int, float],
        sign_map: dict[int, int],
        hill_map: dict[int, int],
        reaction_map: dict[int, tuple[int, ...]],
        transporter_map: dict[int, tuple[int, ...]],
        effector_map: dict[int, tuple[int, ...]],
        m: int,
    ) -> None:
        self._cls = _lib.Proteomics(
            km_map,
            vmax_map,
            sign_map,
            hill_map,
            reaction_map,
            transporter_map,
            effector_map,
            m,
        )

    def get_proteome_params(self, proteomes: list[list[ProteinSpecType]]) -> tuple[
        list[list[float]],
        list[list[float]],
        list[list[list[float]]],
        list[list[list[int]]],
        list[list[list[int]]],
        list[list[list[int]]],
    ]:
        return self._cls.get_proteome_params(proteomes)

    def get_proteome_dict(
        self, proteome: list[ProteinSpecType], molecules: list[str]
    ) -> list[dict[str, Any]]:
        return self._cls.get_proteome_dict(proteome, molecules)


# cells


class Cells:

    def __init__(self, genomics: Genomics, proteomics: Proteomics) -> None:
        self._cls = _lib.Cells(genomics._cls, proteomics._cls)

    def translate_genomes(self, genomes: list[str]) -> tuple[
        list[list[float]],
        list[list[float]],
        list[list[list[float]]],
        list[list[list[int]]],
        list[list[list[int]]],
        list[list[list[int]]],
    ]:
        return self._cls.translate_genomes(genomes)
