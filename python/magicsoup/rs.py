from typing import Any

from magicsoup import _lib  # type: ignore
from magicsoup.constants import ProteinSpecType


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

    def get_proteome_params(
        self, proteomes: list[list[ProteinSpecType]], p: int
    ) -> tuple[
        list[list[float]],
        list[list[float]],
        list[list[list[float]]],
        list[list[list[int]]],
        list[list[list[int]]],
        list[list[list[int]]],
    ]:
        return self._cls.get_proteome_params(proteomes, p)

    def get_proteome_dict(
        self, proteome: list[ProteinSpecType], molecules: list[str]
    ) -> list[dict[str, Any]]:
        return self._cls.get_proteome_dict(proteome, molecules)
