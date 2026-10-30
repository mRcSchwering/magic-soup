//! All tunable parameters. Rates are per bp per generation.
//! Defaults are illustrative for ~1 kb genomes with a total per-genome
//! rate of order 0.01-0.1

#[derive(Clone, Debug)]
pub struct Config {
    // ---- substitutions ----
    /// Per-bp substitution rate.
    pub sub_rate: f64,
    /// Fraction of substitutions that are transitions (A<->G, C<->T).
    /// 1/3 = no bias; ~0.5-0.6 is closer to wild-type E. coli.
    pub transition_prob: f64,

    // ---- small indels (replication slippage) ----
    /// Per-bp indel rate at sites that are NOT inside a tandem repeat.
    pub indel_rate: f64,
    /// Rate multiplier per additional repeat copy: mult = factor^(copies-1).
    pub indel_repeat_factor: f64,
    /// Upper bound of the multiplier (also the rejection-sampling envelope;
    /// candidates are drawn at indel_rate * indel_max_multiplier).
    pub indel_max_multiplier: f64,
    /// Longest repeat unit (bp) recognised as a tandem repeat.
    pub indel_max_unit: usize,
    /// Mean indel size: in bp outside repeats, in repeat units inside repeats.
    pub indel_mean_len: f64,
    /// Hard cap on indel length in bp.
    pub indel_max_len: usize,
    /// Probability that an indel is an insertion (duplication of adjacent sequence).
    pub insertion_prob: f64,

    // ---- homology-dependent rearrangements ----
    /// Per-bp rate of breaks that start a homology search.
    pub break_rate: f64,
    /// Seed length (k) used to find homologous partners.
    pub seed_len: usize,
    /// Minimum total homology (bp) for recombination to be possible.
    pub min_homology: usize,
    /// Homology length (bp) at which a partner is used with probability 1.
    pub saturating_homology: usize,
    /// Minimum length of the segment that is deleted / duplicated / inverted.
    pub min_segment: usize,
    /// Among direct-repeat events: probability of duplication (else deletion).
    pub duplication_prob: f64,

    // ---- genome size limits (events that would violate them are skipped) ----
    pub min_genome_len: usize,
    pub max_genome_len: Option<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            sub_rate: 1e-5,
            transition_prob: 0.55,
            indel_rate: 1e-7,
            indel_repeat_factor: 2.0,
            indel_max_multiplier: 64.0,
            indel_max_unit: 6,
            indel_mean_len: 1.5,
            indel_max_len: 20,
            insertion_prob: 0.5,
            break_rate: 1e-4,
            seed_len: 12,
            min_homology: 15,
            saturating_homology: 60,
            min_segment: 20,
            duplication_prob: 0.5,
            min_genome_len: 50,
            max_genome_len: None,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<(), String> {
        let probs = [
            ("transition_prob", self.transition_prob),
            ("insertion_prob", self.insertion_prob),
            ("duplication_prob", self.duplication_prob),
        ];
        for (name, p) in probs {
            if !(0.0..=1.0).contains(&p) {
                return Err(format!("{name} must be in [0, 1], got {p}"));
            }
        }
        for (name, r) in [
            ("sub_rate", self.sub_rate),
            ("indel_rate", self.indel_rate),
            ("break_rate", self.break_rate),
        ] {
            if !(r >= 0.0 && r.is_finite()) {
                return Err(format!("{name} must be finite and >= 0, got {r}"));
            }
        }
        if self.indel_repeat_factor < 1.0 || self.indel_max_multiplier < 1.0 {
            return Err("indel_repeat_factor and indel_max_multiplier must be >= 1".into());
        }
        if self.indel_mean_len < 1.0 {
            return Err("indel_mean_len must be >= 1".into());
        }
        if self.indel_max_unit == 0 || self.indel_max_len == 0 {
            return Err("indel_max_unit and indel_max_len must be >= 1".into());
        }
        if self.seed_len == 0 {
            return Err("seed_len must be >= 1".into());
        }
        if self.min_homology < self.seed_len {
            return Err("min_homology must be >= seed_len".into());
        }
        if self.saturating_homology < self.min_homology {
            return Err("saturating_homology must be >= min_homology".into());
        }
        Ok(())
    }
}
