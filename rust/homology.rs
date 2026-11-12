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
//!
//! The genome is circular: the seed, the homology search and its extension, and
//! the resulting edit may all cross the seam. The seed is read with wrap, the
//! partner search runs on a once-doubled buffer, extension uses modular indices
//! (capped so the two copies stay disjoint on the ring), and an edit whose
//! region crosses the seam is rotated to the front before the contiguous
//! splice/drain/reverse applies.
use crate::config::MutatorConfig;
use crate::seq::{circular_slice, complement, reverse_complement, reverse_complement_in_place};
use memchr::memmem;
use rand::Rng;
use std::borrow::Cow;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pair {
    /// Direct repeat copies starting at `a < b`, homology length `h`.
    Direct { a: usize, b: usize, h: usize },
    /// Inverted repeat copies starting at `f < r`, homology length `h`.
    Inverted { f: usize, r: usize, h: usize },
}

/// Extend a direct-repeat seed match (copies starting at a and b, initial
/// length k) around the ring. The homology length is capped so the two copies
/// stay disjoint on the circle; extension may cross the seam.
fn extend_direct(s: &[u8], a: usize, b: usize, k: usize) -> (usize, usize, usize) {
    let n = s.len();
    let d = (b + n - a) % n; // forward arc a -> b (the segment A..Y)
    let max_h = d.min(n - d); // disjoint on both arcs of the ring
    let mut a = a;
    let mut b = b;
    let mut h = k.min(max_h);

    // extend left: both copies move back together (segment length d unchanged)
    while h < max_h && s[(a + n - 1) % n] == s[(b + n - 1) % n] {
        a = (a + n - 1) % n;
        b = (b + n - 1) % n;
        h += 1;
    }

    // extend right: grow the shared homology length
    while h < max_h && s[(a + h) % n] == s[(b + h) % n] {
        h += 1;
    }

    (a, b, h)
}

/// Extend an inverted-repeat seed match around the ring.
/// Invariant: s[(f+t) % n] == comp(s[(r+h-1-t) % n]) for t in 0..h.
/// The forward copy grows left and the reverse copy grows right into the outward
/// gap, then both grow into the inward gap; each gap is consumed from both ends
/// (two bases per step) and extension stops before the copies would touch.
fn extend_inverted(s: &[u8], f: usize, r: usize, k: usize) -> (usize, usize, usize) {
    let n = s.len();
    let mut f = f;
    let mut r = r;
    let mut h = k;

    // outward: (f-1) pairs with (r+h)
    loop {
        let outward = (f + n - ((r + h) % n)) % n; // arc (r+h) -> f
        if outward < 2 {
            break;
        }
        let bf = (f + n - 1) % n;
        let br = (r + h) % n;
        if s[bf] != complement(s[br]) {
            break;
        }
        f = bf;
        h += 1;
    }

    // inward: (f+h) pairs with (r-1), keeping the copies separate
    loop {
        let inward = (r + n - ((f + h) % n)) % n; // arc (f+h) -> r
        if inward < 2 {
            break;
        }
        let bf = (f + h) % n;
        let br = (r + n - 1) % n;
        if s[bf] != complement(s[br]) {
            break;
        }
        r = (r + n - 1) % n;
        h += 1;
    }
    (f, r, h)
}

fn homology_weight(h: usize, cfg: &MutatorConfig) -> f64 {
    if h < cfg.min_homology {
        return 0.0;
    }
    let num = (h - cfg.min_homology + 1) as f64;
    let den = (cfg.saturating_homology - cfg.min_homology + 1) as f64;
    (num / den).min(1.0)
}

/// All recombination partners for a break at `brk`, with weights.
fn find_partners(s: &[u8], brk: usize, cfg: &MutatorConfig) -> Vec<(Pair, f64)> {
    let n = s.len();
    let k = cfg.seed_len;
    let seed = circular_slice(s, brk, k); // may wrap the seam
    let rc = reverse_complement(&seed);

    // search a once-doubled buffer so seam-wrapping occurrences are found, then
    // keep one representative per ring position (memchr's SIMD search beats a
    // rolling k-mer hash here)
    let ss = [s, s].concat();
    let direct_hits: Vec<usize> = memmem::find_iter(&ss, &seed).filter(|&j| j < n).collect();
    let inverted_hits: Vec<usize> = memmem::find_iter(&ss, &rc).filter(|&j| j < n).collect();

    let mut out: Vec<(Pair, f64)> = Vec::new();
    for j in direct_hits {
        if j == brk {
            continue;
        }
        let (a, b, h) = extend_direct(s, brk.min(j), brk.max(j), k);
        if (b + n - a) % n < cfg.min_segment {
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
        let gap = (r + n - ((f + h) % n)) % n;
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
fn plan_rearrangement<R: Rng>(seq: &[u8], cfg: &MutatorConfig, rng: &mut R) -> Option<Pair> {
    let n = seq.len();
    let k = cfg.seed_len;
    if n < 2 * k {
        return None;
    }

    let brk = rng.random_range(0..n); // seed may wrap the seam
    let partners = find_partners(seq, brk, cfg);
    if partners.is_empty() {
        return None;
    }

    let total: f64 = partners.iter().map(|(_, w)| w).sum();
    if !rng.random_bool(total.min(1.0)) {
        return None;
    }

    let mut x = rng.random::<f64>() * total;
    for (p, w) in &partners {
        if x < *w {
            return Some(*p);
        }
        x -= w;
    }

    Some(partners[partners.len() - 1].0)
}

/// Apply the recombination product of a chosen repeat pair. A region that runs
/// past the end is rotated to the front first so the contiguous edit applies.
fn apply_rearrangement<R: Rng>(seq: &mut Vec<u8>, pair: Pair, cfg: &MutatorConfig, rng: &mut R) {
    let n = seq.len();
    match pair {
        Pair::Direct { a, b, .. } => {
            let d = (b + n - a) % n; // segment A..Y up to the start of B
            if rng.random_bool(cfg.duplication_prob) {
                if cfg.max_genome_len.is_some_and(|m| n + d > m) {
                    return; // rejected
                }
                if a + d <= n {
                    let segment: Vec<u8> = seq[a..a + d].to_vec();
                    seq.splice((a + d)..(a + d), segment); // duplication
                } else {
                    seq.rotate_left(a);
                    let segment: Vec<u8> = seq[0..d].to_vec();
                    seq.splice(d..d, segment);
                }
            } else {
                if n - d < cfg.min_genome_len {
                    return; // rejected
                }
                if a + d <= n {
                    seq.drain(a..a + d); // deletion
                } else {
                    seq.rotate_left(a);
                    seq.drain(0..d);
                }
            }
        }
        Pair::Inverted { f, r, h } => {
            // rc(A Y B) == A rc(Y) B because B == rc(A)
            let gap = (r + n - ((f + h) % n)) % n; // middle region Y
            let span = 2 * h + gap; // A Y B, always <= n
            if f + span <= n {
                reverse_complement_in_place(&mut seq[f..f + span]); // inversion
            } else {
                seq.rotate_left(f);
                reverse_complement_in_place(&mut seq[0..span]);
            }
        }
    }
}

/// Plan and apply in one call (convenience for use outside the engine).
pub fn try_rearrangement<R: Rng>(seq: &mut Cow<[u8]>, cfg: &MutatorConfig, rng: &mut R) {
    if let Some(pair) = plan_rearrangement(seq, cfg, rng) {
        apply_rearrangement(&mut seq.to_mut(), pair, cfg, rng);
    }
}
