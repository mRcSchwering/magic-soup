//! Single-nucleotide substitutions.

use crate::config::MutatorConfig;
use crate::seq::{base_index, BASES};
use rand::Rng;
use std::borrow::Cow;

/// Replace the base at `pos` with a different base. Returns false (and changes
/// nothing) if the position holds a non-ACGT byte.
pub fn try_substitution<R: Rng>(seq: &mut Cow<[u8]>, pos: usize, cnf: &MutatorConfig, rng: &mut R) {
    let Some(i) = base_index(seq[pos]) else {
        return;
    };

    let j = if rng.random_bool(cnf.transition_prob) {
        i ^ 2 // transition: A<->G, C<->T
    } else if rng.random_bool(0.5) {
        i ^ 1 // transversion
    } else {
        i ^ 3 // the other transversion
    };

    seq.to_mut()[pos] = BASES[j];
}
