use crate::constants::DomainSpecType;
use rayon::prelude::*;
use std::collections::HashMap;

pub fn collect_proteome_idxs(
    proteins: &Vec<Vec<DomainSpecType>>,
    n_prots: &usize,
    n_doms: &usize,
) -> [Vec<Vec<usize>>; 5] {
    let null = 0;
    let empty_seq = vec![null; *n_doms];

    let mut p_dts = vec![];
    let mut p_idxs0 = vec![];
    let mut p_idxs1 = vec![];
    let mut p_idxs2 = vec![];
    let mut p_idxs3 = vec![];
    for doms in proteins {
        let mut d_dts = vec![];
        let mut d_idxs0 = vec![];
        let mut d_idxs1 = vec![];
        let mut d_idxs2 = vec![];
        let mut d_idxs3 = vec![];
        for ([dt, i0, i1, i2, i3], _, _) in doms {
            d_dts.push(*dt);
            d_idxs0.push(*i0);
            d_idxs1.push(*i1);
            d_idxs2.push(*i2);
            d_idxs3.push(*i3);
        }
        let d_fill: Vec<usize> = vec![null; n_doms - d_idxs0.len()];
        d_dts.extend(d_fill.clone());
        d_idxs0.extend(d_fill.clone());
        d_idxs1.extend(d_fill.clone());
        d_idxs2.extend(d_fill.clone());
        d_idxs3.extend(d_fill.clone());
        p_dts.push(d_dts);
        p_idxs0.push(d_idxs0);
        p_idxs1.push(d_idxs1);
        p_idxs2.push(d_idxs2);
        p_idxs3.push(d_idxs3);
    }
    let p_fill = vec![empty_seq.clone(); n_prots - p_idxs0.len()];
    p_dts.extend(p_fill.clone());
    p_idxs0.extend(p_fill.clone());
    p_idxs1.extend(p_fill.clone());
    p_idxs2.extend(p_fill.clone());
    p_idxs3.extend(p_fill.clone());

    [p_dts, p_idxs0, p_idxs1, p_idxs2, p_idxs3]
}

pub fn collect_proteomes_idxs(
    proteomes: &Vec<Vec<Vec<DomainSpecType>>>,
    n_prots: &usize,
) -> [Vec<Vec<Vec<usize>>>; 5] {
    let n_doms: usize = proteomes.iter().flatten().map(|d| d.len()).max().unwrap();

    let prot_specs: Vec<[Vec<Vec<usize>>; 5]> = proteomes
        .into_par_iter()
        .map(|d| collect_proteome_idxs(d, n_prots, &n_doms))
        .collect();

    let mut c_dts = vec![];
    let mut c_idxs0 = vec![];
    let mut c_idxs1 = vec![];
    let mut c_idxs2 = vec![];
    let mut c_idxs3 = vec![];
    for [p_dts, p_idxs0, p_idxs1, p_idxs2, p_idxs3] in prot_specs {
        c_dts.push(p_dts);
        c_idxs0.push(p_idxs0);
        c_idxs1.push(p_idxs1);
        c_idxs2.push(p_idxs2);
        c_idxs3.push(p_idxs3);
    }

    [c_dts, c_idxs0, c_idxs1, c_idxs2, c_idxs3]
}

fn mean(vals: &Vec<f64>, null: f64) -> f64 {
    let n: usize = vals.len();
    if n == 0 {
        return null;
    }
    vals.iter().sum()
}

pub fn calculate_cell_params(
    proteome: &Vec<Vec<DomainSpecType>>,
    n_prots: &usize,
    vmax_map: &HashMap<usize, f64>,
    hill_map: &HashMap<usize, usize>,
    km_map: &HashMap<usize, f64>,
    sign_map: &HashMap<usize, i8>,
    react_map: &HashMap<usize, Vec<isize>>,
    transport_map: &HashMap<usize, Vec<isize>>,
    effector_map: &HashMap<usize, Vec<isize>>,
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
    Vec<Vec<isize>>,
    Vec<Vec<f64>>,
) {
    let mut vmax_p: Vec<f64> = Vec::with_capacity(*n_prots);
    let mut kmn_p: Vec<f64> = Vec::with_capacity(*n_prots);
    let mut n_p: Vec<Vec<usize>> = Vec::with_capacity(*n_prots);
    let mut nf_p: Vec<Vec<usize>> = Vec::with_capacity(*n_prots);
    let mut nb_p: Vec<Vec<usize>> = Vec::with_capacity(*n_prots);
    let mut a_p: Vec<Vec<isize>> = Vec::with_capacity(*n_prots);
    let mut kmr_p: Vec<Vec<f64>> = Vec::with_capacity(*n_prots);

    for protein in proteome {
        // >99% proteins have < 3 domains
        let mut vmax_d: Vec<f64> = Vec::with_capacity(4);
        let mut kmn_d: Vec<f64> = Vec::with_capacity(4);
        let mut n_d: Vec<Vec<usize>> = Vec::with_capacity(4);
        let mut nf_d: Vec<Vec<usize>> = Vec::with_capacity(4);
        let mut nb_d: Vec<Vec<usize>> = Vec::with_capacity(4);
        let mut a_d: Vec<Vec<isize>> = Vec::with_capacity(4);
        let mut kmr_d: Vec<Vec<f64>> = Vec::with_capacity(4);

        for ([dt, i0, i1, i2, i3], _, _) in protein {
            // domain types: 1=catalytic, 2=transporter, 3=regulatory
            // idxs 0-2 are 1-codon indexes used for scalars (n=64 (- stop codons))
            // idx3 is a 2-codon index used for vectors (n=4096 (- stop codons))

            if dt == &1 {
                // Vmax, Km, sign, reacts
                let vmax = vmax_map.get(i0).unwrap();
                let km = km_map.get(i1).unwrap();
                let sign = sign_map.get(i2).unwrap();
                let react = react_map.get(i3).unwrap();
                vmax_d.push(*vmax);
            } else if dt == &2 {
                // Vmax, Km, sign, transport
                let vmax = vmax_map.get(i0).unwrap();
                let km = km_map.get(i1).unwrap();
                let sign = sign_map.get(i2).unwrap();
                let transport = transport_map.get(i3).unwrap();
                vmax_d.push(*vmax);
            } else if dt == &3 {
                // Hill, Km, sign, effector
                let hill = hill_map.get(i0).unwrap();
                let km = km_map.get(i1).unwrap();
                let sign = sign_map.get(i2).unwrap();
                let effector = effector_map.get(i3).unwrap();
                let a: Vec<isize> = effector
                    .iter()
                    .map(|d| d * (*sign as isize) * (*hill as isize))
                    .collect();
                a_d.push(a);
            }
        }
        vmax_p.push(mean(&vmax_d, 0.0));
    }

    (vmax_p, kmn_p, n_p, nf_p, nb_p, a_p, kmr_p)
}

/// return (Vmax_rdy, Kmn, N_rdy, Nf_rdy, Nb_rdy, A_rdy, Kmr)
pub fn calculate_cells_params(
    proteomes: &Vec<Vec<Vec<DomainSpecType>>>,
    n_prots: &usize,
    vmax_map: &HashMap<usize, f64>,
    hill_map: &HashMap<usize, usize>,
    km_map: &HashMap<usize, f64>,
    sign_map: &HashMap<usize, i8>,
    react_map: &HashMap<usize, Vec<isize>>,
    transport_map: &HashMap<usize, Vec<isize>>,
    effector_map: &HashMap<usize, Vec<isize>>,
) -> (
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    Vec<Vec<Vec<usize>>>,
    Vec<Vec<Vec<usize>>>,
    Vec<Vec<Vec<usize>>>,
    Vec<Vec<Vec<isize>>>,
    Vec<Vec<Vec<f64>>>,
) {
    let res: Vec<(
        Vec<f64>,
        Vec<f64>,
        Vec<Vec<usize>>,
        Vec<Vec<usize>>,
        Vec<Vec<usize>>,
        Vec<Vec<isize>>,
        Vec<Vec<f64>>,
    )> = proteomes
        .into_par_iter()
        .map(|d| {
            calculate_cell_params(
                d,
                n_prots,
                vmax_map,
                hill_map,
                km_map,
                sign_map,
                react_map,
                transport_map,
                effector_map,
            )
        })
        .collect();

    let n = res.len();
    let mut Vmax_: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut Kmn_: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut N_: Vec<Vec<Vec<usize>>> = Vec::with_capacity(n);
    let mut Nf_: Vec<Vec<Vec<usize>>> = Vec::with_capacity(n);
    let mut Nb_: Vec<Vec<Vec<usize>>> = Vec::with_capacity(n);
    let mut A_: Vec<Vec<Vec<isize>>> = Vec::with_capacity(n);
    let mut Kmr_: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n);
    for (Vmax, Kmn, N, Nf, Nb, A, Kmr) in res {
        Vmax_.push(Vmax);
        Kmn_.push(Kmn);
        N_.push(N);
        Nf_.push(Nf);
        Nb_.push(Nb);
        A_.push(A);
        Kmr_.push(Kmr);
    }
    (Vmax_, Kmn_, N_, Nf_, Nb_, A_, Kmr_)
}
