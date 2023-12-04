from functools import partial
from numba import jit, prange  # type: ignore
from numba import types as tps  # type: ignore
from numba.typed import List  # pylint: disable=no-name-in-module
from magicsoup.constants import CODON_SIZE, ProteinSpecType

njit = partial(jit, nogil=True, nopython=True, fastmath=True)


@njit
def _reverse_complement(seq: str) -> str:
    rvsd = seq[::-1]
    return (
        rvsd.replace("A", "-")
        .replace("T", "A")
        .replace("-", "T")
        .replace("G", "-")
        .replace("C", "G")
        .replace("-", "C")
    )


@njit
def _get_coding_regions(
    seq: str,
    min_cds_size: int,
    start_codons: list[str],
    stop_codons: list[str],
    is_fwd: bool,
) -> list[tuple[str, int, int, bool]]:
    s = CODON_SIZE
    n = len(seq)
    max_start_idx = n - min_cds_size

    start_idxs = List.empty_list(tps.uint16)
    for start_codon in start_codons:
        i = 0
        while i < max_start_idx:
            try:
                hit = seq[i:].index(start_codon)
                start_idxs.append(i + hit)
                i = i + hit + s
            except Exception:  # only supported in numba
                break

    stop_idxs = List.empty_list(tps.uint16)
    for stop_codon in stop_codons:
        i = 0
        while i < n - s:
            try:
                hit = seq[i:].index(stop_codon)
                stop_idxs.append(i + hit)
                i = i + hit + s
            except Exception:  # only suppprted in numba
                break

    start_idxs.sort()
    stop_idxs.sort()

    by_frame: list[tuple[list[int], ...]] = List()
    for _ in range(3):
        by_frame.append((List.empty_list(tps.uint16), List.empty_list(tps.uint16)))
    for start_idx in start_idxs:
        if start_idx % 3 == 0:
            by_frame[0][0].append(start_idx)
        elif (start_idx + 1) % 3 == 0:
            by_frame[1][0].append(start_idx)
        else:
            by_frame[2][0].append(start_idx)
    for stop_idx in stop_idxs:
        if stop_idx % 3 == 0:
            by_frame[0][1].append(stop_idx)
        elif (stop_idx + 1) % 3 == 0:
            by_frame[1][1].append(stop_idx)
        else:
            by_frame[2][1].append(stop_idx)

    out = List()
    for start_idxs, stop_idxs in by_frame:
        for start_idx in start_idxs:
            new_stop_idxs = List.empty_list(tps.uint16)
            for d in stop_idxs:
                if d > start_idx + s:
                    new_stop_idxs.append(d)
            stop_idxs = new_stop_idxs

            if len(stop_idxs) > 0:
                end_idx = min(stop_idxs) + s
                if end_idx - start_idx > min_cds_size:
                    out.append((seq[start_idx:end_idx], start_idx, end_idx, is_fwd))
            else:
                break

    return out


@njit
def _extract_domains(
    cdss: list[tuple[str, int, int, bool]],
    dom_size: int,
    dom_type_size: int,
    dom_type_map: dict[str, int],
    one_codon_map: dict[str, int],
    two_codon_map: dict[str, int],
) -> list[ProteinSpecType]:
    idx0_slice = slice(0, CODON_SIZE)
    idx1_slice = slice(CODON_SIZE, 2 * CODON_SIZE)
    idx2_slice = slice(2 * CODON_SIZE, 3 * CODON_SIZE)
    idx3_slice = slice(3 * CODON_SIZE, 5 * CODON_SIZE)

    prot_doms = List()
    for cds, cds_start, cds_stop, is_fwd in cdss:
        doms = List()
        is_useful_prot = False

        i = 0
        j = dom_size
        while i + dom_size <= len(cds):
            dom_type_seq = cds[i : i + dom_type_size]
            if dom_type_seq in dom_type_map:
                # 1=catal, 2=trnsp, 3=reg
                dom_type = dom_type_map[dom_type_seq]
                if dom_type != 3:
                    is_useful_prot = True

                dom_spec_seq = cds[i + dom_type_size : i + dom_size]
                idx0 = one_codon_map[dom_spec_seq[idx0_slice]]
                idx1 = one_codon_map[dom_spec_seq[idx1_slice]]
                idx2 = one_codon_map[dom_spec_seq[idx2_slice]]
                idx3 = two_codon_map[dom_spec_seq[idx3_slice]]
                doms.append(((dom_type, idx0, idx1, idx2, idx3), i, i + dom_size))
                i += dom_size
                j += dom_size
            else:
                i += CODON_SIZE
                j += CODON_SIZE

        # protein should have at least 1 non-regulatory domain
        if is_useful_prot:
            prot_doms.append((doms, cds_start, cds_stop, is_fwd))

    return prot_doms


@njit
def _translate_genome(
    genome: str,
    dom_size: int,
    start_codons: list[str],
    stop_codons: list[str],
    dom_type_map: dict[str, int],
    one_codon_map: dict[str, int],
    two_codon_map: dict[str, int],
) -> list[ProteinSpecType]:
    dom_type_size = len(next(iter(dom_type_map)))

    cdsf = _get_coding_regions(
        seq=genome,
        min_cds_size=dom_size,
        start_codons=start_codons,
        stop_codons=stop_codons,
        is_fwd=True,
    )
    bwd = _reverse_complement(genome)
    cdsb = _get_coding_regions(
        seq=bwd,
        min_cds_size=dom_size,
        start_codons=start_codons,
        stop_codons=stop_codons,
        is_fwd=False,
    )

    # concat doesnt work in numba
    for item in cdsb:
        cdsf.append(item)

    prot_doms = _extract_domains(
        cdss=cdsf,
        dom_size=dom_size,
        dom_type_size=dom_type_size,
        dom_type_map=dom_type_map,
        one_codon_map=one_codon_map,
        two_codon_map=two_codon_map,
    )

    return prot_doms


@njit(parallel=True)
def translate_genomes(
    genomes: list[str],
    dom_size: int,
    start_codons: list[str],
    stop_codons: list[str],
    domain_map: dict[str, int],
    one_codon_map: dict[str, int],
    two_codon_map: dict[str, int],
) -> list[list[ProteinSpecType]]:
    n = len(genomes)
    dom_seqs = 0  # testing
    for i in prange(n):  # pylint: disable=not-an-iterable
        genome = genomes[i]
        res = _translate_genome(
            genome,
            dom_size,
            start_codons,
            stop_codons,
            domain_map,
            one_codon_map,
            two_codon_map,
        )
        dom_seqs += len(res)

    return [1, 2]
