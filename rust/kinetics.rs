use crate::constants::DomainSpecType;
use rayon::prelude::*;

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
