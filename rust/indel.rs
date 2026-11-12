//! Small insertions/deletions by replication slippage.
//!
//! A candidate site is drawn uniformly. If it lies in a tandem repeat
//! (homopolymer or short repeat unit), slippage adds or removes whole repeat
//! units, and the site's rate is boosted by `factor^(copies-1)`. Outside
//! repeats, the event is a short tandem duplication or deletion.
//! Rate heterogeneity is implemented by rejection sampling: candidates arrive
//! at `indel_rate * indel_max_multiplier` and are accepted with
//! `multiplier / indel_max_multiplier`.
use crate::config::MutatorConfig;
use crate::seq::geometric_len;
use rand::Rng;
use std::borrow::Cow;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RepeatContext {
    /// Start of the periodic run.
    pub start: usize,
    /// Repeat unit length (bp).
    pub unit: usize,
    /// Number of complete units in the run (1 = not a repeat).
    pub copies: usize,
}

/// Find the longest tandem repeat (unit <= max_unit) containing `i` on the
/// circular genome. Ties are resolved in favour of the shorter unit.
///
/// A repeat may wrap the seam, so periodicity is tested with wrap
/// (`m(j) := s[j % n] == s[(j + u) % n]`) and runs are grown with modular
/// indices. Seeds are taken in the "middle copy" (`i + n`) so a run can extend
/// backward across the seam; the run length is capped at `n` (which also bounds
/// a fully periodic genome and keeps the loops finite).
fn repeat_context(s: &[u8], i: usize, max_unit: usize) -> RepeatContext {
    let n = s.len();
    let mut best = RepeatContext {
        start: i,
        unit: 1,
        copies: 1,
    };
    if n < 2 {
        return best;
    }
    for u in 1..=max_unit.min(n - 1) {
        let m = |j: usize| s[j % n] == s[(j + u) % n];
        // max number of linked positions so the run (count + u bases) <= n
        let cap = n - u;
        // seeds j0 = i - d (mod n), placed near i + n so extension stays >= 0
        for d in 0..=u {
            let j0 = i + n - d;
            if !m(j0) {
                continue;
            }
            let mut p = j0;
            let mut q = j0 + 1; // [p, q) are linked positions
            while (q - (p - 1)) <= cap && m(p - 1) {
                p -= 1;
            }
            while (q + 1 - p) <= cap && m(q) {
                q += 1;
            }
            let copies = (q - p + u) / u;
            if copies > best.copies {
                best = RepeatContext {
                    start: p % n,
                    unit: u,
                    copies,
                };
            }
        }
    }
    best
}

/// A planned indel: duplicate or remove seq[start .. start+len].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Indel {
    pub start: usize,
    pub len: usize,
    pub insert: bool,
}

/// Read-only step: draw a candidate site and decide the event (or reject it).
fn plan_indel<R: Rng>(seq: &[u8], cfg: &MutatorConfig, rng: &mut R) -> Option<Indel> {
    let n = seq.len();
    if n == 0 {
        return None;
    }

    let i = rng.random_range(0..n);
    let ctx = repeat_context(seq, i, cfg.indel_max_unit);

    // rejection step for context-dependent rates
    let mult = if ctx.copies >= 2 {
        cfg.indel_repeat_factor
            .powi((ctx.copies - 1) as i32)
            .min(cfg.indel_max_multiplier)
    } else {
        1.0
    };
    if !rng.random_bool(mult / cfg.indel_max_multiplier) {
        return None;
    }

    let insert = rng.random_bool(cfg.insertion_prob);
    let max_len = cfg.indel_max_len;

    let (start, len) = if ctx.copies >= 2 {
        // slip by k whole units; insertions add at most `copies` units,
        // deletions always leave at least one unit
        let max_units = if insert { ctx.copies } else { ctx.copies - 1 };
        let max_units = max_units.min(max_len / ctx.unit);
        if max_units == 0 {
            return None;
        }
        let k = geometric_len(rng, cfg.indel_mean_len, max_units);
        (ctx.start, k * ctx.unit)
    } else {
        // no end to clamp against on a ring; a short indel may wrap the seam
        let len = geometric_len(rng, cfg.indel_mean_len, max_len);
        (i, len)
    };

    if insert && cfg.max_genome_len.is_some_and(|m| n + len > m) {
        return None;
    }
    if !insert && n - len < cfg.min_genome_len {
        return None;
    }
    Some(Indel { start, len, insert })
}

fn apply_indel(seq: &mut Vec<u8>, e: Indel) {
    let Indel { start, len, insert } = e;
    let n = seq.len();
    // a region that runs past the end is rotated to the front first, so the
    // existing contiguous splice/drain applies (the ring has no fixed origin)
    if start + len > n {
        seq.rotate_left(start);
        if insert {
            let dup: Vec<u8> = seq[0..len].to_vec();
            seq.splice(0..0, dup);
        } else {
            seq.drain(0..len);
        }
    } else if insert {
        // tandem duplication of seq[start .. start+len]
        let dup: Vec<u8> = seq[start..start + len].to_vec();
        seq.splice(start..start, dup); // insertion
    } else {
        seq.drain(start..start + len); // deletion
    }
}

/// Plan and apply in one call.
pub fn try_indel<R: Rng>(seq: &mut Cow<[u8]>, cfg: &MutatorConfig, rng: &mut R) {
    if let Some(e) = plan_indel(seq, cfg, rng) {
        apply_indel(&mut seq.to_mut(), e);
    }
}
