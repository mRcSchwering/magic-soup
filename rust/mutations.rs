use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Poisson};
use rayon::prelude::*;

const NTS: [char; 4] = ['A', 'C', 'T', 'G'];

/// Mutate string with point mutations
/// p mutations per base pair, p_indel chance of indel (vs substitution), p_del chance of deletion (vs insertsion)
/// Returns None if no mutation happened
fn point_mutations(
    mut seq: String,
    idx: usize,
    p: f32,
    p_indel: f32,
    p_del: f32,
) -> Option<(String, usize)> {
    let len = seq.len();
    if len < 1 {
        return None;
    }

    let mut rng = thread_rng();
    let poi = Poisson::new(p * len as f32).expect("Could not init Poisson distribution");
    let mut n_muts = poi.sample(&mut rand::thread_rng()) as usize;
    if n_muts < 1 {
        return None;
    }
    if n_muts > len {
        n_muts = len;
    }

    let mut mut_idxs = rand::seq::index::sample(&mut rng, len, n_muts).into_vec();
    mut_idxs.sort();

    let mut offset: isize = 0;
    for idx in mut_idxs.iter() {
        let rng_idx: usize = (*idx as isize + offset) as usize;
        if rng.gen_bool(p_indel as f64) {
            if rng.gen_bool(p_del as f64) {
                seq.remove(rng_idx);
                offset -= 1;
            } else {
                let nt = NTS.choose(&mut rng);
                if let Some(d) = nt {
                    seq.insert(rng_idx, *d);
                    offset += 1;
                }
            }
        } else {
            let nt = NTS.choose(&mut rng);
            if let Some(d) = nt {
                seq.replace_range(rng_idx..(rng_idx + 1), &d.to_string());
            }
        }
    }

    Some((seq, idx))
}

/// Threaded version of point_mutations for multiple sequences
pub fn point_mutations_threaded(
    seqs: Vec<String>,
    p: f32,
    p_indel: f32,
    p_del: f32,
) -> Vec<(String, usize)> {
    seqs.into_par_iter()
        .enumerate()
        .map(|(i, d)| point_mutations(d, i, p, p_indel, p_del))
        .filter_map(std::convert::identity)
        .collect()
}

/// Recombinate pair of 2 sequences by splitting it with likelihood p per char
/// and rejoining it randomly into 2 sequences.
/// Returns None if no mutation happened.
fn recombinations(
    seq0: String,
    seq1: String,
    idx: usize,
    p: f32,
) -> Option<(String, String, usize)> {
    let n0 = seq0.len();
    let n1 = seq1.len();
    let n_both = &n0 + &n1;
    if n_both < 1 {
        return None;
    }

    let poi = Poisson::new(p * n_both as f32).expect("Could not init Poisson distribution");
    let mut n_muts = poi.sample(&mut rand::thread_rng()) as usize;
    if n_muts < 1 {
        return None;
    }
    if n_muts > n_both {
        n_muts = n_both;
    }

    let mut rng = thread_rng();
    let mut mut_idxs = rand::seq::index::sample(&mut rng, n_both, n_muts).into_vec();
    mut_idxs.sort();

    // idxs are distributed according to seq lens
    let total_len = n0 + n1;
    let mut idxs: (Vec<usize>, Vec<usize>) = (
        Vec::with_capacity(n_muts * n0 / total_len),
        Vec::with_capacity(n_muts * n1 / total_len),
    );
    for mut_idx in mut_idxs {
        if mut_idx >= n0 {
            idxs.1.push(mut_idx - n0);
        } else {
            idxs.0.push(mut_idx);
        }
    }

    // for each strand: n strand breaks + 1
    let mut parts: Vec<&str> = Vec::with_capacity(n_muts + 2);

    let mut i = 0;
    for j in idxs.0 {
        parts.push(&seq0[i..j]);
        i = j;
    }
    parts.push(&seq0[i..]);

    i = 0;
    for j in idxs.1 {
        parts.push(&seq1[i..j]);
        i = j;
    }
    parts.push(&seq1[i..]);

    parts.shuffle(&mut thread_rng());
    let s = rng.gen_range(0..parts.len());
    let new_0 = parts[..s].join("");
    let new_1 = parts[s..].join("");

    Some((new_0, new_1, idx))
}

/// Threaded version of recombinations for multiple sequence pairs
pub fn recombinations_threaded(
    seq_pairs: Vec<(String, String)>,
    p: f32,
) -> Vec<(String, String, usize)> {
    seq_pairs
        .into_par_iter()
        .enumerate()
        .map(|(i, d)| recombinations(d.0, d.1, i, p))
        .filter_map(std::convert::identity)
        .collect()
}
