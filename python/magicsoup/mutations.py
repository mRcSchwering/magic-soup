import random

from magicsoup import rs


class Mutator:

    def __init__(
        self,
        seed: float | None = None,
        sub_rate: float = 1e-9,
        transition_prob: float = 0.55,
        indel_rate: float = 1e-7,
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

    def mutate(self, genomes: list[str]) -> list[tuple[int, str]]:
        # TODO: could return as numpy -> tensors
        return self.rs.mutate(genomes=genomes)

    def _setup_rs(self) -> None:
        self.rs = rs.Mutator(
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
