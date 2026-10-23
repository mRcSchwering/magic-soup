from typing import Any

from magicsoup import _lib  # type: ignore
from magicsoup.constants import ProteinSpecType


class _ReduceMixin:
    """
    Instead of trying to pickle rust instances re-initialize them
    when un-pickling (__reduce__). Therefore, save __init__ values
    (_state). Must be tuple so keep order!
    """

    _state: tuple

    def __reduce__(self):
        return (self.__class__, self._state)


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


class Genomics(_ReduceMixin):

    def __init__(
        self,
        start_codons: list[str],
        stop_codons: list[str],
        domain_type_2_token: dict[str, int],
        one_codon_2_token: dict[str, int],
        two_codon_2_token: dict[str, int],
        dom_size_nt: int,
        dom_type_size_nt: int,
    ) -> None:
        self._cls = _lib.Genomics(
            start_codons,
            stop_codons,
            domain_type_2_token,
            one_codon_2_token,
            two_codon_2_token,
            dom_size_nt,
            dom_type_size_nt,
        )
        self._state = (
            start_codons,
            stop_codons,
            domain_type_2_token,
            one_codon_2_token,
            two_codon_2_token,
            dom_size_nt,
            dom_type_size_nt,
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


class Proteomics(_ReduceMixin):

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
        self._state = (
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


class Cells(_ReduceMixin):

    def __init__(self, genomics: Genomics, proteomics: Proteomics) -> None:
        self._cls = _lib.Cells(genomics._cls, proteomics._cls)
        self._state = (genomics, proteomics)

    def translate_genomes(self, genomes: list[str]) -> tuple[
        list[list[float]],
        list[list[float]],
        list[list[list[float]]],
        list[list[list[int]]],
        list[list[list[int]]],
        list[list[list[int]]],
    ]:
        return self._cls.translate_genomes(genomes)
