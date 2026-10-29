from typing import TYPE_CHECKING

import numpy as np

from magicsoup import _lib  # type: ignore

if TYPE_CHECKING:
    from magicsoup.genomics import ProteinSpec


class _ReduceMixin:
    """
    Instead of trying to pickle rust instances re-initialize them
    when un-pickling (__reduce__). Therefore, save __init__ values
    (_state). Must be tuple so keep order!
    """

    _state: tuple

    def __reduce__(self):
        return (self.__class__, self._state)


# mutations


def point_mutations(
    seqs: list[str], p: float, p_indel: float, p_del: float
) -> list[tuple[str, int]]:
    return _lib.point_mutations(seqs, p, p_indel, p_del)


def recombinations(
    seq_pairs: list[tuple[str, str]], p: float
) -> list[tuple[str, str, int]]:
    return _lib.recombinations(seq_pairs, p)


# genomics


def reverse_complement(seq: str) -> str:
    return _lib.reverse_complement(seq)


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
            [d.encode() for d in start_codons],
            [d.encode() for d in stop_codons],
            {k.encode(): v for k, v in domain_type_2_token.items()},
            {k.encode(): v for k, v in one_codon_2_token.items()},
            {k.encode(): v for k, v in two_codon_2_token.items()},
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

    def translate_genomes(self, genomes: list[str]) -> "list[list[ProteinSpec]]":
        # NOTE: even cheaper is it to send all genomes as one long buffer
        # buffer = "".join(strarr)
        # offsets = np.array([0] + list(np.cumsum([len(g) for g in strarr])), dtype=np.int64)
        #       then in rust:
        # let offs = offsets.as_slice()?;
        # let slices: Vec<&[u8]> = offs.windows(2).map(|w| &buf[w[0] as usize..w[1] as usize]).collect();
        # revcomp_many(&slices).map_err(|(i, b)| {PyValueError::new_err("genome invalid base")})
        return self._cls.translate_genomes(genomes)

    def translate_genomes_arr(
        self, genomes: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        # NOTE: even cheaper is it to send all genomes as one long buffer
        # buffer = "".join(strarr)
        # offsets = np.array([0] + list(np.cumsum([len(g) for g in strarr])), dtype=np.int64)
        #       then in rust:
        # let offs = offsets.as_slice()?;
        # let slices: Vec<&[u8]> = offs.windows(2).map(|w| &buf[w[0] as usize..w[1] as usize]).collect();
        # revcomp_many(&slices).map_err(|(i, b)| {PyValueError::new_err("genome invalid base")})
        return self._cls.translate_genomes_arr(genomes)

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
    ) -> "list[ProteinSpec]":
        return self._cls.extract_domains(genome, cdss)
