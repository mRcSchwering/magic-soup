//! Small sequence helpers. Genomes are ASCII byte strings of A, C, G, T.
//! Other bytes are tolerated: they are never substituted and are kept
//! unchanged by complementation.

pub const BASES: [u8; 4] = *b"ACGT";

/// A=0, C=1, G=2, T=3. With this order, the transition partner of base i is
/// i ^ 2 (A<->G, C<->T) and the two transversion partners are i ^ 1 and i ^ 3.
#[inline]
pub fn base_index(b: u8) -> Option<usize> {
    match b {
        b'A' => Some(0),
        b'C' => Some(1),
        b'G' => Some(2),
        b'T' => Some(3),
        _ => None,
    }
}

#[inline]
pub fn complement(b: u8) -> u8 {
    match b {
        b'A' => b'T',
        b'T' => b'A',
        b'C' => b'G',
        b'G' => b'C',
        other => other,
    }
}

pub fn reverse_complement(s: &[u8]) -> Vec<u8> {
    s.iter().rev().map(|&b| complement(b)).collect()
}

/// Reverse-complement a slice in place.
pub fn reverse_complement_in_place(s: &mut [u8]) {
    s.reverse();
    for b in s.iter_mut() {
        *b = complement(*b);
    }
}

/// Geometric length >= 1 with the given mean, truncated at `max`.
pub fn geometric_len<R: rand::Rng>(rng: &mut R, mean: f64, max: usize) -> usize {
    let p_continue = 1.0 - 1.0 / mean; // mean of 1 + Geom = 1/p_stop
    let mut len = 1;
    while len < max && rng.random_bool(p_continue) {
        len += 1;
    }
    len
}
