//! Single-nucleotide substitutions.

use crate::seq::{base_index, BASES};
use rand::Rng;

/// Replace the base at `pos` with a different base. Returns false (and changes
/// nothing) if the position holds a non-ACGT byte.
pub fn substitute<R: Rng>(seq: &mut [u8], pos: usize, transition_prob: f64, rng: &mut R) -> bool {
    let Some(i) = base_index(seq[pos]) else {
        return false;
    };
    let j = if rng.random_bool(transition_prob) {
        i ^ 2 // transition: A<->G, C<->T
    } else if rng.random_bool(0.5) {
        i ^ 1 // transversion
    } else {
        i ^ 3 // the other transversion
    };
    seq[pos] = BASES[j];
    true
}
