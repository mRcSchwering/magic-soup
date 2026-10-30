//! Homology-dependent rearrangements (RecA-like).
//!
//! 1. A break occurs at a random position; the k-mer there is the seed.
//! 2. Every other occurrence of the seed (direct repeat) and of its reverse
//!    complement (inverted repeat) is a candidate partner.
//! 3. Each match is extended to its full homology length h and weighted by
//!    w(h) = clamp((h - h_min + 1) / (h_sat - h_min + 1), 0, 1).
//! 4. Recombination happens with probability min(1, sum w); the partner is
//!    chosen proportional to w. Otherwise the break is repaired faithfully.
//! 5. The outcome follows from repeat orientation:
//!    direct   X A Y B Z -> deletion      X B Z
//!                       -> duplication   X A Y A Y B Z
//!    inverted X A Y B Z -> inversion     X A rc(Y) B Z   (B = rc(A))

use crate::config::Config;
use crate::seq::{complement, reverse_complement, reverse_complement_in_place};
use memchr::memmem;
use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pair {
    /// Direct repeat copies starting at `a < b`, homology length `h`.
    Direct { a: usize, b: usize, h: usize },
    /// Inverted repeat copies starting at `f < r`, homology length `h`.
    Inverted { f: usize, r: usize, h: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RearrangementOutcome {
    /// No homologous partner found (break repaired from the sister copy).
    NoPartner,
    /// Partners found but recombination not used, or a size limit blocked it.
    Rejected,
    Deletion {
        len: usize,
    },
    Duplication {
        len: usize,
    },
    Inversion {
        len: usize,
    },
}

/// Extend a direct-repeat seed match (copies at a < b, initial length k).
pub fn extend_direct(s: &[u8], mut a: usize, mut b: usize, k: usize) -> (usize, usize, usize) {
    let n = s.len();
    let mut h = k;
    while a > 0 && s[a - 1] == s[b - 1] {
        a -= 1;
        b -= 1;
        h += 1;
    }
    while b + h < n && a + h < b && s[a + h] == s[b + h] {
        h += 1;
    }
    (a, b, h)
}

/// Extend an inverted-repeat seed match. Invariant: s[f+t] == comp(s[r+h-1-t]).
pub fn extend_inverted(s: &[u8], mut f: usize, mut r: usize, k: usize) -> (usize, usize, usize) {
    let n = s.len();
    let mut h = k;
    // outward: f-1 pairs with r+h
    while f > 0 && r + h < n && s[f - 1] == complement(s[r + h]) {
        f -= 1;
        h += 1;
    }
    // inward: f+h pairs with r-1, keeping the copies separate
    while f + h + 1 < r && s[f + h] == complement(s[r - 1]) {
        r -= 1;
        h += 1;
    }
    (f, r, h)
}

fn homology_weight(h: usize, cfg: &Config) -> f64 {
    if h < cfg.min_homology {
        return 0.0;
    }
    let num = (h - cfg.min_homology + 1) as f64;
    let den = (cfg.saturating_homology - cfg.min_homology + 1) as f64;
    (num / den).min(1.0)
}

/// All recombination partners for a break at `brk`, with weights.
pub fn find_partners(s: &[u8], brk: usize, cfg: &Config) -> Vec<(Pair, f64)> {
    let k = cfg.seed_len;
    let seed = &s[brk..brk + k];
    let rc = reverse_complement(seed);
    // memchr's SIMD substring search is faster here than a rolling k-mer hash
    let direct_hits = memmem::find_iter(s, seed);
    let inverted_hits: Vec<usize> = memmem::find_iter(s, &rc).collect();

    let mut out: Vec<(Pair, f64)> = Vec::new();
    for j in direct_hits {
        if j == brk {
            continue;
        }
        let (a, b, h) = extend_direct(s, brk.min(j), brk.max(j), k);
        if b - a < cfg.min_segment {
            continue;
        }
        let w = homology_weight(h, cfg);
        let pair = Pair::Direct { a, b, h };
        // the same repeat pair can be reached from several seeds; dedupe
        if w > 0.0 && !out.iter().any(|(p, _)| *p == pair) {
            out.push((pair, w));
        }
    }
    for j in inverted_hits {
        let (lo, hi) = (brk.min(j), brk.max(j));
        if hi < lo + k {
            continue; // overlapping (palindromic) seed: no separate copy
        }
        let (f, r, h) = extend_inverted(s, lo, hi, k);
        let gap = r - (f + h);
        if gap < cfg.min_segment {
            continue;
        }
        let w = homology_weight(h, cfg);
        let pair = Pair::Inverted { f, r, h };
        if w > 0.0 && !out.iter().any(|(p, _)| *p == pair) {
            out.push((pair, w));
        }
    }
    out
}

/// Read-only step: draw a break and decide whether and with which partner it
/// recombines. `Err` carries the outcome when nothing will change.
pub fn plan_rearrangement<R: Rng>(
    seq: &[u8],
    cfg: &Config,
    rng: &mut R,
) -> Result<Pair, RearrangementOutcome> {
    let n = seq.len();
    let k = cfg.seed_len;
    if n < 2 * k {
        return Err(RearrangementOutcome::NoPartner);
    }
    let brk = rng.random_range(0..=n - k);
    let partners = find_partners(seq, brk, cfg);
    if partners.is_empty() {
        return Err(RearrangementOutcome::NoPartner);
    }
    let total: f64 = partners.iter().map(|(_, w)| w).sum();
    if !rng.random_bool(total.min(1.0)) {
        return Err(RearrangementOutcome::Rejected);
    }
    let mut x = rng.random::<f64>() * total;
    for (p, w) in &partners {
        if x < *w {
            return Ok(*p);
        }
        x -= w;
    }
    Ok(partners[partners.len() - 1].0)
}

/// Plan and apply in one call (convenience for use outside the engine).
pub fn try_rearrangement<R: Rng>(
    seq: &mut Vec<u8>,
    cfg: &Config,
    rng: &mut R,
) -> RearrangementOutcome {
    match plan_rearrangement(seq, cfg, rng) {
        Ok(pair) => apply(seq, pair, cfg, rng),
        Err(outcome) => outcome,
    }
}

/// Apply the recombination product of a chosen repeat pair.
pub fn apply<R: Rng>(
    seq: &mut Vec<u8>,
    pair: Pair,
    cfg: &Config,
    rng: &mut R,
) -> RearrangementOutcome {
    let n = seq.len();
    match pair {
        Pair::Direct { a, b, .. } => {
            let d = b - a; // segment A..Y up to the start of B
            if rng.random_bool(cfg.duplication_prob) {
                if cfg.max_genome_len.is_some_and(|m| n + d > m) {
                    return RearrangementOutcome::Rejected;
                }
                let segment: Vec<u8> = seq[a..b].to_vec();
                seq.splice(b..b, segment);
                RearrangementOutcome::Duplication { len: d }
            } else {
                if n - d < cfg.min_genome_len {
                    return RearrangementOutcome::Rejected;
                }
                seq.drain(a..b);
                RearrangementOutcome::Deletion { len: d }
            }
        }
        Pair::Inverted { f, r, h } => {
            // rc(A Y B) == A rc(Y) B because B == rc(A)
            reverse_complement_in_place(&mut seq[f..r + h]);
            RearrangementOutcome::Inversion { len: r - (f + h) }
        }
    }
}
