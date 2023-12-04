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


# TODO: how to get rid of warning?
@njit(parallel=True)
def _point_mutations_string_list(
    seqs: list[str], p: float, p_indel: float
) -> np.ndarray:
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


_NT_INTS = (65, 67, 84, 71)  # A, C, T, G


# TODO: how to get rid of warning?
@njit(parallel=True)
def _point_mutations_int_list(
    arrs: list[np.ndarray], p: float, p_indel: float
) -> np.ndarray:
    n = len(arrs)

    chgnd_idxs = np.full((n,), -1, dtype=np.int32)
    for arr_i in prange(n):  # pylint: disable=not-an-iterable
        lst = [d for d in arrs[arr_i]]
        lst_len = len(lst)
        mut_idxs = [i for i in range(lst_len - 1, -1, -1) if _bernoulli(p)]
        if len(mut_idxs) < 1:
            continue

        chgnd_idxs[arr_i] = arr_i
        indels = np.random.binomial(n=1, p=p_indel, size=len(mut_idxs))

        for idx, is_indel in zip(mut_idxs, indels):
            if is_indel:
                is_del = random.random() > 0.333
                if is_del:
                    del lst[idx]
                else:
                    nti = random.randint(0, len(_NT_INTS) - 1)
                    lst.insert(idx, _NT_INTS[nti])
            else:
                nti = random.randint(0, len(_NT_INTS) - 1)
                lst[idx] = _NT_INTS[nti]

        arrs[arr_i] = np.array(lst, dtype=np.uint8)

    return chgnd_idxs[chgnd_idxs > -1]


def point_mutations_string_list(
    seqs: list[str], p=1e-6, p_indel=0.4
) -> list[tuple[str, int]]:
    seqs_lst = List(seqs)
    idxs = _point_mutations_string_list(seqs_lst, p, p_indel)
    seqs = list(seqs_lst)
    return [(seqs[i], i) for i in idxs]


def point_mutations_int_list(
    seqs: list[str], p=1e-6, p_indel=0.4
) -> list[tuple[str, int]]:
    lst = List([np.frombuffer(d.encode(), dtype=np.uint8) for d in seqs])
    res = _point_mutations_int_list(lst, p, p_indel)
    seqs = [d.tostring().decode() for d in lst]  # type: ignore
    return [(seqs[i], i) for i in res]


def point_mutations_int_list_raw(
    seqs: list[str], p=1e-6, p_indel=0.4
) -> list[tuple[str, int]]:
    lst = List([np.frombuffer(d.encode(), dtype=np.uint8) for d in seqs])
    # res = _point_mutations_int_list(lst, p, p_indel)
    seqs = [d.tostring().decode() for d in lst]  # type: ignore
    return [(seqs[i], i) for i in range(100)]
