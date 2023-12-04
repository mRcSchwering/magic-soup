import random
from functools import partial
import numpy as np
from numba import jit, prange  # type: ignore
from numba import types as tps  # type: ignore
from numba.typed import List  # pylint: disable=no-name-in-module
from magicsoup.constants import ALL_NTS

njit = partial(jit, nogil=True, nopython=True, fastmath=True)


@njit(tps.boolean(tps.float32))
def _bernoulli(p: float) -> bool:
    return random.random() < p


@njit(tps.unicode_type(tps.unicode_type, tps.uint32))
def _substitution(seq: str, idx: int) -> str:
    nti = random.randint(0, len(ALL_NTS) - 1)
    return seq[:idx] + ALL_NTS[nti] + seq[idx + 1 :]


@njit(tps.unicode_type(tps.unicode_type, tps.uint32))
def _indel(seq: str, idx: int) -> str:
    sample = random.randint(0, 2)
    if sample < 2:  # 1:3 likelihood
        return seq[:idx] + seq[idx + 1 :]
    nti = random.randint(0, len(ALL_NTS) - 1)
    return seq[:idx] + ALL_NTS[nti] + seq[idx:]


@njit(parallel=True)
def _point_mutations(seqs: list[str], p: float, p_indel: float) -> list[int]:
    n = len(seqs)
    if n == 0:
        return List([d for d in range(0)])

    lens = [len(d) for d in seqs]
    s_max = max(lens)

    mask = np.zeros((n, s_max), dtype=np.uint16)
    for i, s in enumerate(lens):
        mask[i, :s] = 1

    muts = np.random.binomial(n=1, p=p, size=(n, s_max))
    mut_idxs = np.argwhere(muts * mask)

    n_muts = len(mut_idxs)
    indels = np.random.binomial(n=1, p=p_indel, size=(n_muts))

    for (seq_i, pos_i), is_indel in zip(mut_idxs, indels):
        if bool(is_indel):
            seqs[seq_i] = _indel(seq=seqs[seq_i], idx=pos_i)
        else:
            seqs[seq_i] = _substitution(seq=seqs[seq_i], idx=pos_i)

    return List(set([d[0].item() for d in mut_idxs]))


# TODO: how to get rid of warning?
@njit(parallel=True)
def _point_mutations2(seqs: list[str], p: float, p_indel: float) -> np.ndarray:
    n = len(seqs)

    chgnd_idxs = np.full((n,), -1, dtype=np.int32)
    for seq_i in prange(n):  # pylint: disable=not-an-iterable
        seq = seqs[seq_i]
        seq_len = len(seq)
        mut_idxs = [i for i in range(seq_len - 1, -1, -1) if _bernoulli(p)]
        n_muts = len(mut_idxs)
        if n_muts > 0:
            chgnd_idxs[seq_i] = seq_i
            indels = [_bernoulli(p_indel) for _ in range(n_muts)]
            for nt_i, is_indel in zip(mut_idxs, indels):
                if is_indel:
                    seq = _indel(seq=seq, idx=nt_i)
                else:
                    seq = _substitution(seq=seq, idx=nt_i)
            seqs[seq_i] = seq

    return chgnd_idxs[chgnd_idxs > -1]


def point_mutations(seqs: list[str], p=1e-6, p_indel=0.4) -> list[tuple[str, int]]:
    seqs_lst = List(seqs)
    idxs = _point_mutations(seqs_lst, p, p_indel)
    seqs = list(seqs)
    return [(seqs[i], i) for i in idxs]


def point_mutations2(seqs: list[str], p=1e-6, p_indel=0.4) -> list[tuple[str, int]]:
    seqs_lst = List(seqs)
    idxs = _point_mutations2(seqs_lst, p, p_indel)
    seqs = list(seqs_lst)
    return [(seqs[i], i) for i in idxs]
