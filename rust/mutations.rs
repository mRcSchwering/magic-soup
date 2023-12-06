use rand::{Rng, thread_rng};
use rand::seq::SliceRandom;
use rand_distr::{Poisson, Distribution};
use rayon::prelude::*;

const NTS: [char; 4] = ['A', 'C', 'T', 'G'];

/// Consume string and mutate it with point mutations
/// p mutations per nucleotide, p_indel chance of indel (vs substitution), p_del chance of deletion (vs insertsion)
/// Returns None if no mutation happened
pub fn point_mutate_seq(rng: &mut rand::prelude::ThreadRng, mut seq: String, p: f64, p_indel: f64, p_del: f64) -> Option<String> {
    let len = seq.len();

    let poi = Poisson::new(p * len as f64).expect("Could not init Poisson distribution");
    let mut n_muts = poi.sample(&mut rand::thread_rng()) as usize;
    if n_muts < 1 {
        return None
    }
    if n_muts > len {
        n_muts = len;
    }
    
    let mut_idxs = rand::seq::index::sample(rng, len, n_muts);

    let mut offset: isize = 0;
    for idx in mut_idxs {
        if rng.gen_bool(p_indel) {
            if rng.gen_bool(p_del) {
                seq.remove((idx as isize + offset) as usize);
                offset -= 1;
            } else {
                let nt = NTS.choose(rng);
                if let Some(d) = nt {
                    seq.insert((idx as isize + offset) as usize, *d);
                    offset += 1;
                }
            }
        } else {
            let nt = NTS.choose(rng);
            if let Some(d) = nt {
                seq.replace_range(idx..(idx + 1),&d.to_string());
            }
        }
    }
    
    Some(seq)
}


/// Threaded version of point_mutate_seq for multiple sequences
pub fn point_mutate_seqs(seqs: Vec<String>, p: f64, p_indel: f64, p_del: f64) -> Vec<(String, usize)> {
    let par_iter = seqs.into_par_iter().enumerate().map(|(idx, seq)| {
        let mut rng = thread_rng();
        let opt = point_mutate_seq(&mut rng, seq, p, p_indel, p_del);
        if let Some(d) = opt {
            return Some((d, idx))
        } else {
            return None
        }
    });
    let res: Vec<(String, usize)> = par_iter.filter_map(std::convert::identity).collect();
    res
}