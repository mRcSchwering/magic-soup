import random
import re
from collections import Counter

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


def test_indel_slippage_on_circular_tandem_repeat() -> None:
    # the whole ring is one tandem repeat, so every replication-slippage event -
    # including those that straddle the seam - adds or removes whole repeat units
    unit = "ACGTT"
    genomes = StrArray.from_strings([unit * 40 for _ in range(2000)])
    mutator = Mutator(sub_rate=0.0, indel_rate=5e-3, break_rate=0.0, min_genome_len=30)

    n_changed = 0
    for _ in range(5):
        idxs, mutated = mutator.mutate(genomes=genomes)
        for i, seq in zip(idxs.tolist(), mutated.tolist()):
            assert set(seq) <= set("ACGT")
            delta = len(seq) - len(genomes[i])
            assert delta % len(unit) == 0  # whole units only, also across the seam
            if delta != 0:
                n_changed += 1
            genomes[i] = seq

    assert n_changed > 0


def test_homology_rearrangement_across_seam() -> None:
    # a 60 bp direct repeat with one copy wrapping the seam; the segment between
    # the two copies has length d, so a duplication grows the genome by d and a
    # deletion shrinks it by d (the chosen recombination product spans the seam)
    repeat = random_genome(60)
    mid1 = random_genome(40)
    mid2 = random_genome(40)
    k = 30
    genome = repeat[k:] + mid1 + repeat + mid2 + repeat[:k]  # copy 1 wraps the seam
    n = len(genome)
    copy2_start = len(repeat[k:]) + len(mid1)
    copy1_start = n - k
    d = copy1_start - copy2_start

    def make_mutator(duplication_prob: float) -> Mutator:
        return Mutator(
            seed=1,
            sub_rate=0.0,
            indel_rate=0.0,
            break_rate=5e-3,
            seed_len=12,
            min_homology=15,
            saturating_homology=60,
            min_segment=20,
            duplication_prob=duplication_prob,
            min_genome_len=30,
            max_genome_len=10_000,
        )

    genomes = StrArray.from_strings([genome for _ in range(4000)])
    _, dup = make_mutator(duplication_prob=1.0).mutate(genomes=genomes)
    assert all(set(s) <= set("ACGT") for s in dup.tolist())
    dup_deltas = Counter(len(s) - n for s in dup.tolist() if len(s) != n)
    assert dup_deltas.most_common(1)[0][0] == d

    genomes = StrArray.from_strings([genome for _ in range(4000)])
    _, dele = make_mutator(duplication_prob=0.0).mutate(genomes=genomes)
    assert all(set(s) <= set("ACGT") for s in dele.tolist())
    del_deltas = Counter(len(s) - n for s in dele.tolist() if len(s) != n)
    assert del_deltas.most_common(1)[0][0] == -d


@pytest.mark.slow
def test_mutations_preserve_validity_under_seam_pressure(rng: random.Random) -> None:
    # hammer repeat-heavy genomes with every mechanism at a high rate so many
    # events fall on the seam; outputs must stay valid and within size bounds
    lo, hi = 30, 2000
    mutator = Mutator(
        sub_rate=1e-3,
        indel_rate=1e-3,
        break_rate=1e-3,
        min_genome_len=lo,
        max_genome_len=hi,
    )

    pool: list[str] = []
    for _ in range(2000):
        kind = rng.random()
        if kind < 0.4:
            pool.append(random_genome(rng.randint(100, 500)))
        elif kind < 0.7:
            unit = random_genome(rng.randint(4, 12))
            pool.append((unit * 100)[: rng.randint(100, 500)])  # tandem repeats
        else:
            arm = random_genome(rng.randint(30, 60))
            pool.append(arm + random_genome(rng.randint(30, 80)) + arm + random_genome(40))
    genomes = StrArray.from_strings(pool)

    n_changed = 0
    for _ in range(20):
        idxs, mutated = mutator.mutate(genomes=genomes)
        for i, seq in zip(idxs.tolist(), mutated.tolist()):
            assert set(seq) <= set("ACGT")
            assert lo <= len(seq) <= hi
            genomes[i] = seq
        n_changed += len(idxs)

    assert n_changed > 0
