import random
from typing import TypedDict

import numpy as np

from magicsoup.core import rs
from magicsoup.core.strarray import A, StrArray


class MutatorKwargs(TypedDict, total=False):
    seed: float | None
    sub_rate: float
    transition_prob: float
    indel_rate: float
    indel_repeat_factor: float
    indel_max_multiplier: float
    indel_max_unit: int
    indel_mean_len: float
    indel_max_len: int
    insertion_prob: float
    break_rate: float
    seed_len: int
    min_homology: int
    saturating_homology: int
    min_segment: int
    duplication_prob: float
    min_genome_len: int
    max_genome_len: int | None


class Mutator:
    """
    Mutator with substitutions, small indels, and homology-dependent re-arrangements

    Arguments:
        sub_rate: per-bp substitution rate
        transition_prob: probability that a substitutions is a transitions (transversion otherwise)
        indel_rate: per-bp indel rate at sites that are not inside a tandem repeat
        indel_repeat_factor: rate multiplier per additional repeat copy
        indel_max_multiplier: upper bound of the multiplier
        indel_max_unit: longest repeat unit (bp) recognised as a tandem repeat
        indel_mean_len: mean indel size in bp outside repeats, in repeat units inside repeats
        indel_max_len: hard cap on indel length in bp
        insertion_prob: probability that an indel is an insertion (deletion otherwise)
        break_rate: per-bp rate of breaks that start a homology search
        seed_len: seed length used to find homologous partners
        min_homology: minimum total homology (bp) for recombination to be possible
        saturating_homology: homology length (bp) at which a partner is used with probability 1
        min_segment: minimum length of the segment that is deleted / duplicated / inverted
        duplication_prob: probability of duplication among direct-repeat events (else deletion)
        min_genome_len: minimum acceptable genome size
        max_genome_len: maximum acceptable genome size

    First substitutions are applied, then indels, then homology-dependent re-arrangements.
    Any mutation event that would create a genome smaller `min_genome_len`
    or a genome larger `max_genome_len` is cancelled.

    Transitions are A<->G or C<->T substitutions.
    Transversions are other substitutions (e.g. A<->C).
    For any bp there is one possible transition but two possible transversions.
    Each of the two possible transversions has 50% likelihood.
    So `transition_prob=1/3` would define un-biased substitutions.

    Indel candidate locations are drawn with per-bp rate `indel_rate * indel_max_multiplier`.
    If a candidate location is inside a tandem repeat, slippage is simulated with
    probability `factor / indel_max_multiplier` where `factor = indel_repeat_factor^(copies-1)` (clamped to `indel_max_multiplier`).
    `copies` is the number of repeat units found in the candidate location.
    A number of repeat units drawn from a geometric distribution `indel_mean_len`
    is then inserted or removed based on `insertion_prob`.
    Whether a location is inside a tandem repeat is checked using `indel_max_unit`.
    Outside of tandem repeat locations simple short indels are simulated.
    A candidate receives an indel with probability `1 / indel_max_multiplier`.
    A number of base pairs drawn from a geometric distribution with `indel_mean_len`
    are then inserted or removed based on `insertion_prob`.
    In any case the maximum indel size (bp) is always `indel_max_len`.

    Candidates for homology-dependent re-arrangements (RecA-like) are introduced by breaks
    with per-bp rate `break_rate`. The k-mer (`seed_len`) at this position becomes the seed.
    Every other occurence of the seed and its reverse-complement is a potential partner.
    Each match is extended to its full homology length `h` and weighted by
    `w = (h - min_homology + 1) / (saturating_homology - min_homology + 1)` (clamped to 0 and 1).
    Recombination happens with probability `sum(w)`, otherwise the break is repaired correctly.
    The partner is chosen proportional to `w` and the outcome depends on orientation.
    For direct matches `X A Y B Z` is deleted to `X B Z` or inserted to `X A Y A Y B Z` depending
    on `duplication_prob`. For inverted matches `X A Y B Z` is inverted to `X A rc(Y) B Z`.
    """

    def __init__(
        self,
        seed: float | None = None,
        sub_rate: float = 1e-5,
        transition_prob: float = 0.55,
        indel_rate: float = 1e-6,
        indel_repeat_factor: float = 2.0,
        indel_max_multiplier: float = 64.0,
        indel_max_unit: int = 6,
        indel_mean_len: float = 1.5,
        indel_max_len: int = 20,
        insertion_prob: float = 0.5,
        break_rate: float = 1e-4,
        seed_len: int = 12,
        min_homology: int = 15,
        saturating_homology: int = 60,
        min_segment: int = 20,
        duplication_prob: float = 0.5,
        min_genome_len: int = 50,
        max_genome_len: int | None = None,
    ) -> None:
        self.seed = seed
        self.sub_rate = sub_rate
        self.transition_prob = transition_prob
        self.indel_rate = indel_rate
        self.indel_repeat_factor = indel_repeat_factor
        self.indel_max_multiplier = indel_max_multiplier
        self.indel_max_unit = indel_max_unit
        self.indel_mean_len = indel_mean_len
        self.indel_max_len = indel_max_len
        self.insertion_prob = insertion_prob
        self.break_rate = break_rate
        self.seed_len = seed_len
        self.min_homology = min_homology
        self.saturating_homology = saturating_homology
        self.min_segment = min_segment
        self.duplication_prob = duplication_prob
        self.min_genome_len = min_genome_len
        self.max_genome_len = max_genome_len

        # setup rust class
        self._setup_rs()

    def mutate(self, genomes: A) -> tuple[np.ndarray, A]:
        """Apply mutations"""
        idxs, seqs = self._rs.mutate(*genomes.as_buffers())
        return idxs, StrArray.from_strings(seqs)

    def _setup_rs(self) -> None:
        self._rs = rs.Mutator(
            seed=self.seed or random.getrandbits(8),
            sub_rate=self.sub_rate,
            transition_prob=self.transition_prob,
            indel_rate=self.indel_rate,
            indel_repeat_factor=self.indel_repeat_factor,
            indel_max_multiplier=self.indel_max_multiplier,
            indel_max_unit=self.indel_max_unit,
            indel_mean_len=self.indel_mean_len,
            indel_max_len=self.indel_max_len,
            insertion_prob=self.insertion_prob,
            break_rate=self.break_rate,
            seed_len=self.seed_len,
            min_homology=self.min_homology,
            saturating_homology=self.saturating_homology,
            min_segment=self.min_segment,
            duplication_prob=self.duplication_prob,
            min_genome_len=self.min_genome_len,
            max_genome_len=self.max_genome_len,
        )
