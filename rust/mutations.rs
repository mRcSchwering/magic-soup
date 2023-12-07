use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Poisson};
use rayon::prelude::*;

const NTS: [char; 4] = ['A', 'C', 'T', 'G'];

/// Mutate string with point mutations
/// p mutations per nucleotide, p_indel chance of indel (vs substitution), p_del chance of deletion (vs insertsion)
/// Returns None if no mutation happened
fn point_mutate_seq(
    mut seq: String,
    idx: usize,
    p: f64,
    p_indel: f64,
    p_del: f64,
) -> Option<(String, usize)> {
    let len = seq.len();
    if len < 1 {
        return None;
    }

    let mut rng = thread_rng();
    let poi = Poisson::new(p * len as f64).expect("Could not init Poisson distribution");
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
        if rng.gen_bool(p_indel) {
            if rng.gen_bool(p_del) {
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

/// Threaded version of point_mutate_seq for multiple sequences
pub fn point_mutate_seqs(
    seqs: Vec<String>,
    p: f64,
    p_indel: f64,
    p_del: f64,
) -> Vec<(String, usize)> {
    seqs.into_par_iter()
        .enumerate()
        .map(|(i, d)| point_mutate_seq(d, i, p, p_indel, p_del))
        .filter_map(std::convert::identity)
        .collect()
}
