//! Batch driver: distributes events over all genomes and applies them.
//!
//! Instead of testing every bp, the number of events of each kind in the whole
//! population is drawn from a Poisson distribution (total_bp * rate) and each
//! event is assigned to a genome with probability proportional to its length
//! (binary search on prefix sums). Only genomes that receive events are copied.
//!
//! Per genome, events are applied in the order substitutions -> indels ->
//! rearrangements. Substitution positions are drawn on the original sequence;
//! indel and break positions are drawn on the current sequence when applied.

use crate::config::Config;
use crate::homology::try_rearrangement;
use crate::indel::try_indel;
use crate::substitutions::try_substitution;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Poisson};
use std::borrow::Cow;

#[derive(Clone, Copy)]
enum Kind {
    Sub(usize), // offset in the original genome
    Indel,
    Break,
}

pub struct Mutator {
    pub cfg: Config,
    rng: SmallRng,
}

impl Mutator {
    pub fn new(cfg: Config, seed: u64) -> Result<Self, String> {
        cfg.validate()?;
        Ok(Mutator {
            cfg,
            rng: SmallRng::seed_from_u64(seed),
        })
    }

    fn poisson(&mut self, lambda: f64) -> u64 {
        if lambda <= 0.0 {
            return 0;
        }
        Poisson::new(lambda)
            .map(|d| d.sample(&mut self.rng) as u64)
            .unwrap_or(0)
    }

    /// Mutate one generation. Returns (index, new sequence) for every genome
    /// that actually changed; unchanged genomes are not copied or returned.
    pub fn mutate<S: AsRef<[u8]>>(&mut self, genomes: &[S]) -> Vec<(usize, Vec<u8>)> {
        // prefix[i] = end offset of genome i in the virtual concatenation
        let mut prefix = Vec::with_capacity(genomes.len());

        let mut total: u64 = 0;
        for g in genomes {
            total += g.as_ref().len() as u64;
            prefix.push(total);
        }

        if total == 0 {
            return Vec::new();
        }

        let n_sub = self.poisson(total as f64 * self.cfg.sub_rate);
        let n_indel =
            self.poisson(total as f64 * self.cfg.indel_rate * self.cfg.indel_max_multiplier);
        let n_break = self.poisson(total as f64 * self.cfg.break_rate);
        let mut events: Vec<(usize, Kind)> =
            Vec::with_capacity((n_sub + n_indel + n_break) as usize);

        for (count, kind) in [(n_sub, 0u8), (n_indel, 1), (n_break, 2)] {
            for _ in 0..count {
                let x = self.rng.random_range(0..total);
                let g = prefix.partition_point(|&end| end <= x);
                let start = if g == 0 { 0 } else { prefix[g - 1] };
                let k = match kind {
                    0 => Kind::Sub((x - start) as usize),
                    1 => Kind::Indel,
                    _ => Kind::Break,
                };
                events.push((g, k));
            }
        }

        // stable sort keeps substitutions before indels before breaks
        events.sort_by_key(|&(g, _)| g);

        let mut out = Vec::new();
        let mut i = 0;

        while i < events.len() {
            let g = events[i].0;
            let mut j = i;

            while j < events.len() && events[j].0 == g {
                j += 1;
            }

            // copy-on-write: the genome is only cloned once an event changes it
            let mut seq = Cow::Borrowed(genomes[g].as_ref());
            for &(_, kind) in &events[i..j] {
                apply_event(&mut seq, kind, &self.cfg, &mut self.rng);
            }

            if let Cow::Owned(v) = seq {
                out.push((g, v));
            }

            i = j;
        }
        out
    }
}

fn apply_event(seq: &mut Cow<[u8]>, kind: Kind, cfg: &Config, rng: &mut SmallRng) {
    match kind {
        Kind::Sub(pos) => try_substitution(seq, pos, cfg, rng),
        Kind::Indel => try_indel(seq, cfg, rng),
        Kind::Break => try_rearrangement(seq, cfg, rng),
    }
}
