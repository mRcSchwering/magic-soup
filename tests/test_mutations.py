import re

import edlib  # type: ignore
import pytest
from magicsoup.core.strarray import StrArray
from magicsoup.core.util import random_genome
from magicsoup.mutations import Mutator

_CIGAR_RE = re.compile(r"(\d+)([=XID])")


def _sub_indel_counts(a: str, b: str):
    res = edlib.align(a, b, mode="NW", task="path")
    cigar = res["cigar"] or ""
    c = {"=": 0, "X": 0, "I": 0, "D": 0}
    for n, op in _CIGAR_RE.findall(cigar):
        c[op] += int(n)

    return {
        "subs": c["X"],
        "indels": c["I"] + c["D"],
    }


def test_reasonable_mutation_rates() -> None:
    mutator = Mutator()

    n = 10_000
    genomes = StrArray.from_strings([random_genome() for _ in range(n)])
    idxs, mutated = mutator.mutate(genomes=genomes)

    # expect about 30 per 10k genomes
    exp_rate = 0.003
    assert abs(len(mutated) / n - exp_rate) < 0.1

    # all mutated sequences are different
    assert all(genomes[i] != d for i, d in zip(idxs, mutated))

    # no bullshit was added
    assert all(set(d) <= set("ACTG") for d in mutated)

    # about 75% are substitutions
    subs = 0
    indels = 0
    for idx, seq in zip(idxs, mutated):
        muts = _sub_indel_counts(genomes[idx], seq)
        subs += muts["subs"]
        indels += muts["indels"]

    total = subs + indels
    p_subs = subs / total
    p_indels = indels / total
    assert p_subs == pytest.approx(0.75, abs=0.2)
    assert p_indels == pytest.approx(0.25, abs=0.2)


@pytest.mark.slow
@pytest.mark.parametrize(
    "sub, indel, homo",
    [
        (1e-3, 0.0, 0.0),
        (0.0, 1e-3, 0.0),
        (0.0, 0.0, 1e-3),
    ],
)
def test_mutation_mechanisms(sub: float, indel: float, homo: float) -> None:
    mutator = Mutator(sub_rate=sub, indel_rate=indel, break_rate=homo)

    n = 10_000
    genomes = StrArray.from_strings([random_genome(10) * 100 for _ in range(n)])
    idxs, mutated = mutator.mutate(genomes=genomes)
    assert len(mutated) > 0

    # mutated sequences are different (homol can still create non-mutated ones)
    assert sum(genomes[i] != d for i, d in zip(idxs, mutated)) / len(mutated) > 0.9

    # no bullshit was added
    assert all(set(d) <= set("ACTG") for d in mutated)
