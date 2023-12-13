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

/// Calculate cell params for one cell
/// return (Vmax_rdy, Kmn, N_rdy, Nf_rdy, Nb_rdy, A_rdy, Kmr)
pub fn calculate_cell_params(
    proteome: &Vec<Vec<DomainSpecType>>,
    n_prots: &usize,
    n_mols: &usize,
    vmax_map: &HashMap<usize, f64>,
    hill_map: &HashMap<usize, usize>,
    km_map: &HashMap<usize, f64>,
    sign_map: &HashMap<usize, bool>,
    react_map: &HashMap<usize, Vec<(usize, isize)>>,
    transport_map: &HashMap<usize, Vec<(usize, isize)>>,
    effector_map: &HashMap<usize, Vec<(usize, usize)>>,
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<Vec<isize>>,
    Vec<Vec<isize>>,
    Vec<Vec<isize>>,
    Vec<Vec<isize>>,
    Vec<Vec<f64>>,
) {
    let n_signals = 2 * n_mols;
    let mut vmax_p: Vec<f64> = Vec::with_capacity(*n_prots);
    let mut kmn_p: Vec<f64> = Vec::with_capacity(*n_prots);
    let mut n_p: Vec<Vec<isize>> = Vec::with_capacity(*n_prots);
    let mut nf_p: Vec<Vec<isize>> = Vec::with_capacity(*n_prots);
    let mut nb_p: Vec<Vec<isize>> = Vec::with_capacity(*n_prots);
    let mut a_p: Vec<Vec<isize>> = Vec::with_capacity(*n_prots);
    let mut kmr_p: Vec<Vec<f64>> = Vec::with_capacity(*n_prots);

    for protein in proteome {
        // >99% proteins have < 3 domains

        // running Vmax sum and Vmax couter
        let mut vmax_d: f64 = 0.0;
        let mut n_vmax_d: usize = 0;

        // running Km sum and Km counter
        let mut kmn_d: f64 = 0.0;
        let mut n_kmn_d: usize = 0;

        // running N, Nf and Nb sums
        let mut n_d: Vec<isize> = vec![0; n_signals];
        let mut nf_d: Vec<isize> = vec![0; n_signals];
        let mut nb_d: Vec<isize> = vec![0; n_signals];

        // running A
        let mut a_d: Vec<isize> = vec![0; n_signals];

        // running Kmr sum and counter for each signal
        let mut kmr_d: Vec<f64> = vec![0.0; n_signals];
        let mut n_effectors_d: Vec<(usize, usize)> = vec![];

        for ([dt, i0, i1, i2, i3], _, _) in protein {
            // domain types: 1=catalytic, 2=transporter, 3=regulatory
            // idxs 0-2 are 1-codon indexes used for scalars (n=64 (- stop codons))
            // idx3 is a 2-codon index used for vectors (n=4096 (- stop codons))

            if dt == &1 {
                // Vmax, Km, sign, reacts
                let vmax = vmax_map.get(i0).unwrap();
                let km = km_map.get(i1).unwrap();
                let is_fwd = sign_map.get(i2).unwrap();
                let reacts = react_map.get(i3).unwrap();

                // increment Vmax, Kmn
                vmax_d += vmax;
                n_vmax_d += 1;
                kmn_d += km;
                n_kmn_d += 1;

                // increment N, Nf, Nb for each signal
                if *is_fwd {
                    for (mol, number) in reacts {
                        n_d[*mol] += number;
                        if number < &0 {
                            nf_d[*mol] -= number;
                        } else {
                            nb_d[*mol] += number;
                        }
                    }
                } else {
                    for (mol, number) in reacts {
                        n_d[*mol] -= number;
                        if number < &0 {
                            nb_d[*mol] -= number;
                        } else {
                            nf_d[*mol] += number;
                        }
                    }
                }
            } else if dt == &2 {
                // Vmax, Km, sign, transport
                let vmax = vmax_map.get(i0).unwrap();
                let km = km_map.get(i1).unwrap();
                let is_fwd = sign_map.get(i2).unwrap();
                let transports = transport_map.get(i3).unwrap();

                // increment Vmax, Kmn
                vmax_d += vmax;
                n_vmax_d += 1;
                kmn_d += km;
                n_kmn_d += 1;

                // increment N, Nf, Nb for each signal
                if *is_fwd {
                    for (mol, number) in transports {
                        n_d[*mol] += number;
                        if number < &0 {
                            nf_d[*mol] -= number;
                        } else {
                            nb_d[*mol] += number;
                        }
                    }
                } else {
                    for (mol, number) in transports {
                        n_d[*mol] -= number;
                        if number < &0 {
                            nb_d[*mol] -= number;
                        } else {
                            nf_d[*mol] += number;
                        }
                    }
                }
            } else if dt == &3 {
                // Hill, Km, sign, effector
                let hill = hill_map.get(i0).unwrap(); // [1;5]
                let km = km_map.get(i1).unwrap(); // [1e-2;100]
                let is_fwd = sign_map.get(i2).unwrap(); // {-1;1}
                let effectors = effector_map.get(i3).unwrap(); // {0;1} len n_signals

                // increment Kmr, A for each signal
                if *is_fwd {
                    for (mol, number) in effectors {
                        a_d[*mol] += (number * hill) as isize;
                        kmr_d[*mol] += km;
                        n_effectors_d.push((*mol, *number));
                    }
                } else {
                    for (mol, number) in effectors {
                        a_d[*mol] -= (number * hill) as isize;
                        kmr_d[*mol] += km;
                        n_effectors_d.push((*mol, *number));
                    }
                }
            }
        }

        // divide Vmax and Km sum to get avg
        vmax_d /= n_vmax_d as f64;
        kmn_d /= n_kmn_d as f64;

        // divide Kmr sum for each signal to get avgs
        for (effector, number) in n_effectors_d {
            kmr_d[effector] /= number as f64;
        }

        // push protein
        vmax_p.push(vmax_d);
        kmn_p.push(kmn_d);
        n_p.push(n_d);
        nf_p.push(nf_d);
        nb_p.push(nb_d);
        a_p.push(a_d);
        kmr_p.push(kmr_d);
    }

    // fill up empty vectors
    let rest = n_prots - proteome.len();
    vmax_p.extend(vec![0.0; rest]);
    kmn_p.extend(vec![0.0; rest]);
    n_p.extend(vec![vec![0; n_signals]; rest]);
    nf_p.extend(vec![vec![0; n_signals]; rest]);
    nb_p.extend(vec![vec![0; n_signals]; rest]);
    a_p.extend(vec![vec![0; n_signals]; rest]);
    kmr_p.extend(vec![vec![0.0; n_signals]; rest]);

    (vmax_p, kmn_p, n_p, nf_p, nb_p, a_p, kmr_p)
}

/// Calculate cell params for many cells
/// return (Vmax_rdy, Kmn, N_rdy, Nf_rdy, Nb_rdy, A_rdy, Kmr)
pub fn calculate_cell_params_threaded(
    proteomes: &Vec<Vec<Vec<DomainSpecType>>>,
    n_prots: &usize,
    n_mols: &usize,
    vmax_map: &HashMap<usize, f64>,
    hill_map: &HashMap<usize, usize>,
    km_map: &HashMap<usize, f64>,
    sign_map: &HashMap<usize, bool>,
    react_map: &HashMap<usize, Vec<(usize, isize)>>,
    transport_map: &HashMap<usize, Vec<(usize, isize)>>,
    effector_map: &HashMap<usize, Vec<(usize, usize)>>,
) -> (
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    Vec<Vec<Vec<isize>>>,
    Vec<Vec<Vec<isize>>>,
    Vec<Vec<Vec<isize>>>,
    Vec<Vec<Vec<isize>>>,
    Vec<Vec<Vec<f64>>>,
) {
    let res: Vec<(
        Vec<f64>,
        Vec<f64>,
        Vec<Vec<isize>>,
        Vec<Vec<isize>>,
        Vec<Vec<isize>>,
        Vec<Vec<isize>>,
        Vec<Vec<f64>>,
    )> = proteomes
        .into_par_iter()
        .map(|d| {
            calculate_cell_params(
                d,
                n_prots,
                n_mols,
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
    let mut vmax_c: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut kmn_c: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut n_c: Vec<Vec<Vec<isize>>> = Vec::with_capacity(n);
    let mut nf_c: Vec<Vec<Vec<isize>>> = Vec::with_capacity(n);
    let mut nb_c: Vec<Vec<Vec<isize>>> = Vec::with_capacity(n);
    let mut a_c: Vec<Vec<Vec<isize>>> = Vec::with_capacity(n);
    let mut kmr_c: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n);
    for (vmax_p, kmn_p, n_p, nf_p, nb_p, a_p, kmr_p) in res {
        vmax_c.push(vmax_p);
        kmn_c.push(kmn_p);
        n_c.push(n_p);
        nf_c.push(nf_p);
        nb_c.push(nb_p);
        a_c.push(a_p);
        kmr_c.push(kmr_p);
    }
    (vmax_c, kmn_c, n_c, nf_c, nb_c, a_c, kmr_c)
}
