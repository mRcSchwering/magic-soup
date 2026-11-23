import math
from collections import Counter
from collections.abc import Iterable

import magicsoup as ms
from magicsoup.core.config import GAS_CONSTANT, ROOM_TEMP


def _parse_cigar(cigar: str) -> list[tuple[str, int]]:
    ops = []
    n = 0
    for ch in cigar:
        if ch.isdigit():
            n += int(ch)
        else:
            ops.append((ch, int(n)))
            n = 0
    return ops


def ani_from_cigar(cigar: str):
    """
    Get Average Nucleotide Identity (ANI) from a
    Concise Idiosyncratic Gapped Alignment Report (CIGAR)
    """
    m = 0
    mm = 0
    for op, n in _parse_cigar(cigar):
        if op == "=":
            m += n
        elif op == "X":
            mm += n
    aln = m + mm
    return m / aln


def _kmers(seq: str, k: int) -> set[str]:
    seq = seq.upper()
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def jaccard(seqa: str, seqb: str, kmer: int = 21) -> float:
    """Get Jaccard similarity"""
    a = _kmers(seqa, k=kmer)
    b = _kmers(seqb, k=kmer)
    return len(a & b) / len(a | b)


def ani_from_jaccard(jacc: float, kmer: int = 21) -> float:
    """
    Get Average Nucleotide Identity (ANI)
    from Jaccard similarity using Mash equation.
    """
    if jacc <= 0:
        return 0.0  # no shared k-mers
    d = -math.log(2 * jacc / (1 + jacc)) / kmer  # Mash distance
    return 1 - d


def _get_kf_kb(energy: float, km: float) -> tuple[float, float]:
    k_e = math.exp(-energy / ROOM_TEMP / GAS_CONSTANT)
    return (km / math.sqrt(k_e), km * math.sqrt(k_e))


def set_trnsp(world: ms.World, c: int, p: int, mol: ms.Molecule, km: float, vmax: float) -> None:
    """Set transporter domain for molecule in cell c protein p with Km, Vmax"""
    cells = world.cells
    molecules = world.chemistry.molecules
    m2i = world.chemistry.mol_2_idx
    k_f, k_b = _get_kf_kb(0, km)
    cells._v_max[c, p] = vmax
    cells._k_f[c, p] = k_f
    cells._k_b[c, p] = k_b
    cells._N_f[c, p, m2i[mol]] = 1
    cells._N_b[c, p, m2i[mol] + len(molecules)] = 1
    N = (world.cells._N_b[c, p] - world.cells._N_f[c, p]).to(world.ftype)
    cells._N_e[c, p] = N * cells.proteomics._cv_ratio


def set_catal(
    world: ms.World,
    c: int,
    p: int,
    prods: Iterable[ms.Molecule],
    subs: Iterable[ms.Molecule],
    km: float,
    vmax: float,
) -> None:
    """Set catalytic domain for reaction in cell c protein p with Km, Vmax"""
    cells = world.cells
    energy = sum(d.energy for d in prods) - sum(d.energy for d in subs)
    fwd_cntr = Counter(subs)
    bwd_cntr = Counter(prods)
    k_f, k_b = _get_kf_kb(energy, km)
    cells._v_max[c, p] = vmax
    cells._k_f[c, p] = k_f
    cells._k_b[c, p] = k_b
    for mol, n in fwd_cntr.items():
        cells._N_f[c, p, world.chemistry.mol_2_idx[mol]] = n
    for mol, n in bwd_cntr.items():
        cells._N_b[c, p, world.chemistry.mol_2_idx[mol]] = n
    N = (world.cells._N_b[c, p] - world.cells._N_f[c, p]).to(world.ftype)
    cells._N_e[c, p] = N * cells.proteomics._cv_ratio


def set_regul(world: ms.World, c: int, p: int, kr: float, mol: ms.Molecule, act: bool) -> None:
    """Set regulatory domain for molecule in cell c protein p with Kr"""
    cells = world.cells
    sign = 1 if act else -1
    cells._k_r[c, p] = kr
    cells._N_h[c, p, world.chemistry.mol_2_idx[mol]] = sign
