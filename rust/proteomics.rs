use pyo3::marker::Python;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};
use rayon::prelude::*;
use std::collections::HashMap;

pub type DomainSpecType = ((u8, u8, u8, u8, u16), usize, usize);
pub type ProteinSpecType = (Vec<DomainSpecType>, usize, usize, bool);
pub type ProteinParamsType = (f32, f32, Vec<f32>, Vec<i8>, Vec<i8>, Vec<i8>);

// Get domain v_max, k_m, sign, reaction values from indices
fn get_cat_dom_params(
    domain: &DomainSpecType,
    vmax_map: &HashMap<u8, f32>,
    k_m_map: &HashMap<u8, f32>,
    sign_map: &HashMap<u8, i8>,
    react_map: &HashMap<u16, Vec<i8>>,
) -> (f32, f32, i8, Vec<i8>) {
    let v_max = vmax_map.get(&domain.0 .1).expect("Incomplete vmax_map");
    let k_m = k_m_map.get(&domain.0 .2).expect("Incomplete k_m_map");
    let sign = sign_map.get(&domain.0 .3).expect("Incomplete sign_map");
    let react = react_map.get(&domain.0 .4).expect("Incomplete react_map");
    (*v_max, *k_m, *sign, react.clone())
}

// Get domain v_max, k_m, sign, transporter from indices
fn get_tsp_dom_params(
    domain: &DomainSpecType,
    vmax_map: &HashMap<u8, f32>,
    k_m_map: &HashMap<u8, f32>,
    sign_map: &HashMap<u8, i8>,
    transp_map: &HashMap<u16, Vec<i8>>,
) -> (f32, f32, i8, Vec<i8>) {
    let v_max = vmax_map.get(&domain.0 .1).expect("Incomplete vmax_map");
    let k_m = k_m_map.get(&domain.0 .2).expect("Incomplete k_m_map");
    let sign = sign_map.get(&domain.0 .3).expect("Incomplete sign_map");
    let transp = transp_map.get(&domain.0 .4).expect("Incomplete transp_map");
    (*v_max, *k_m, *sign, transp.clone())
}

// Get protein v_max, k_m, n_f, n_b from domain values
fn agg_cat_tsp_params(
    params: &Vec<(f32, f32, i8, Vec<i8>)>,
    m: &usize,
) -> (f32, f32, Vec<i8>, Vec<i8>) {
    let n_doms = params.len() as f32;

    // average v_max over domains
    let mut v_max_sum = 0 as f32;
    // average k_m over domains
    let mut k_m_sum = 0 as f32;

    // sum n_f, n_b over domains
    let mut n_f = vec![0 as i8; *m];
    let mut n_b = vec![0 as i8; *m];

    if n_doms == 0.0 {
        return (0.0, 0.0, n_f, n_b);
    }

    for param in params.iter() {
        v_max_sum += param.0;
        k_m_sum += param.1;

        // if fwd
        if param.2 > 0 {
            for (i, &val) in param.3.iter().enumerate() {
                if val > 0 {
                    // positive n is created
                    n_b[i] += val;
                } else {
                    // negative n is destroyed
                    n_f[i] -= val;
                }
            }
        // if bwd
        } else {
            for (i, &val) in param.3.iter().enumerate() {
                if val > 0 {
                    // positive n is destroyed
                    n_f[i] += val;
                } else {
                    // negative n is created
                    n_b[i] -= val;
                }
            }
        }
    }

    let v_max = v_max_sum / n_doms;
    let k_m = k_m_sum / n_doms;

    (v_max, k_m, n_f, n_b)
}

// Get domain hill, k_m, sign, effector from indices
fn get_reg_dom_params(
    domain: &DomainSpecType,
    hill_map: &HashMap<u8, i8>,
    k_m_map: &HashMap<u8, f32>,
    sign_map: &HashMap<u8, i8>,
    effect_map: &HashMap<u16, Vec<i8>>,
) -> (i8, f32, i8, Vec<i8>) {
    let hill = hill_map.get(&domain.0 .1).expect("Incomplete hill_map");
    let k_m = k_m_map.get(&domain.0 .2).expect("Incomplete k_m_map");
    let sign = sign_map.get(&domain.0 .3).expect("Incomplete sign_map");
    let effect = effect_map.get(&domain.0 .4).expect("Incomplete effect_map");
    (*hill, *k_m, *sign, effect.clone())
}

// Get protein k_r, n_h from domain values
fn agg_reg_params(params: &Vec<(i8, f32, i8, Vec<i8>)>, m: &usize) -> (Vec<f32>, Vec<i8>) {
    // per molecule average k_r over domains
    let mut k_r_sum = vec![0 as f32; *m];
    let mut k_r_len = vec![0 as f32; *m];

    // sum n_h over domains
    let mut n_h = vec![0 as i8; *m];

    for param in params.iter() {
        for (i, &val) in param.3.iter().enumerate() {
            if val != 0 {
                k_r_sum[i] += param.1;
                k_r_len[i] += 1.0;

                // positive hill for activating, negative for inhibiting
                n_h[i] += param.2 * param.0;
            }
        }
    }

    let k_r: Vec<f32> = k_r_sum
        .iter()
        .zip(k_r_len.iter())
        .map(|(&sum, &len)| if len > 0.0 { sum / len } else { 0.0 })
        .collect();

    (k_r, n_h)
}

// Get protein parameters from protein specification
fn get_protein_params(
    protein_spec: &ProteinSpecType,
    k_m_map: &HashMap<u8, f32>,
    vmax_map: &HashMap<u8, f32>,
    sign_map: &HashMap<u8, i8>,
    hill_map: &HashMap<u8, i8>,
    reaction_map: &HashMap<u16, Vec<i8>>,
    transporter_map: &HashMap<u16, Vec<i8>>,
    effector_map: &HashMap<u16, Vec<i8>>,
    m: &usize,
) -> ProteinParamsType {
    let mut cat_tsp_dom_params: Vec<(f32, f32, i8, Vec<i8>)> = Vec::new();
    let mut reg_dom_params: Vec<(i8, f32, i8, Vec<i8>)> = Vec::new();

    for domain_spec in protein_spec.0.iter() {
        // collect domains by type
        // 1=catalytic, 2=transporter, 3=regulatory
        if domain_spec.0 .0 == 1 {
            cat_tsp_dom_params.push(get_cat_dom_params(
                &domain_spec,
                &vmax_map,
                &k_m_map,
                &sign_map,
                &reaction_map,
            ));
        } else if domain_spec.0 .0 == 2 {
            cat_tsp_dom_params.push(get_tsp_dom_params(
                &domain_spec,
                &vmax_map,
                &k_m_map,
                &sign_map,
                &transporter_map,
            ));
        } else if domain_spec.0 .0 == 3 {
            reg_dom_params.push(get_reg_dom_params(
                &domain_spec,
                &hill_map,
                &k_m_map,
                &sign_map,
                &effector_map,
            ));
        }
    }

    let (v_max, k_m, n_f, n_b) = agg_cat_tsp_params(&cat_tsp_dom_params, &m);
    let (k_r, n_h) = agg_reg_params(&reg_dom_params, &m);

    (v_max, k_m, k_r, n_f, n_b, n_h)
}

// Get proteome parameters from proteome specifications
fn get_proteome_params(
    proteome_spec: &Vec<ProteinSpecType>,
    k_m_map: &HashMap<u8, f32>,
    vmax_map: &HashMap<u8, f32>,
    sign_map: &HashMap<u8, i8>,
    hill_map: &HashMap<u8, i8>,
    reaction_map: &HashMap<u16, Vec<i8>>,
    transporter_map: &HashMap<u16, Vec<i8>>,
    effector_map: &HashMap<u16, Vec<i8>>,
    m: &usize,
) -> Vec<ProteinParamsType> {
    proteome_spec
        .iter()
        .map(|protein_spec| {
            get_protein_params(
                protein_spec,
                k_m_map,
                vmax_map,
                sign_map,
                hill_map,
                reaction_map,
                transporter_map,
                effector_map,
                m,
            )
        })
        .collect()
}

// Threaded version of get_proteome_params() for multiple proteomes
pub fn get_proteome_params_threaded(
    proteomes_spec: &Vec<Vec<ProteinSpecType>>,
    k_m_map: &HashMap<u8, f32>,
    vmax_map: &HashMap<u8, f32>,
    sign_map: &HashMap<u8, i8>,
    hill_map: &HashMap<u8, i8>,
    reaction_map: &HashMap<u16, Vec<i8>>,
    transporter_map: &HashMap<u16, Vec<i8>>,
    effector_map: &HashMap<u16, Vec<i8>>,
    m: &usize,
) -> Vec<Vec<ProteinParamsType>> {
    proteomes_spec
        .into_par_iter()
        .map(|d| {
            get_proteome_params(
                d,
                k_m_map,
                vmax_map,
                sign_map,
                hill_map,
                reaction_map,
                transporter_map,
                effector_map,
                m,
            )
        })
        .collect()
}

// Proteome Python Representation

// translate domain type int to char
fn get_domtype_char(domtype: &u8) -> char {
    match domtype {
        1 => 'C',
        2 => 'T',
        3 => 'R',
        _ => ' ',
    }
}

// set domain specification for catalytic domain on PyDict
fn set_catalytic_domain_dict(
    kwargs: &Bound<PyDict>,
    molecules: &Vec<String>,
    domain_spec: &DomainSpecType,
    vmax_map: &HashMap<u8, f32>,
    k_m_map: &HashMap<u8, f32>,
    sign_map: &HashMap<u8, i8>,
    react_map: &HashMap<u16, Vec<i8>>,
) {
    let (v_max, k_m, sign, react) =
        get_cat_dom_params(&domain_spec, &vmax_map, &k_m_map, &sign_map, &react_map);

    // must be at least 1 for each
    let mut lfts: Vec<String> = Vec::with_capacity(2);
    let mut rgts: Vec<String> = Vec::with_capacity(2);
    for (mol_i, n) in react.iter().enumerate() {
        let signed_n = sign * *n;
        if signed_n == 0 {
            continue;
        } else if signed_n > 0 {
            let mol = &molecules[mol_i];
            rgts.extend((0..n.abs()).map(|_| mol.to_string()));
        } else {
            let mol = &molecules[mol_i];
            lfts.extend((0..n.abs()).map(|_| mol.to_string()));
        }
    }
    kwargs.set_item("km", k_m).unwrap();
    kwargs.set_item("vmax", v_max).unwrap();
    kwargs.set_item("reaction", (lfts, rgts)).unwrap();
}

// set domain specification for transporter domain on PyDict
fn set_transporter_domain_dict(
    kwargs: &Bound<PyDict>,
    molecules: &Vec<String>,
    domain_spec: &DomainSpecType,
    vmax_map: &HashMap<u8, f32>,
    k_m_map: &HashMap<u8, f32>,
    sign_map: &HashMap<u8, i8>,
    transp_map: &HashMap<u16, Vec<i8>>,
) {
    let (v_max, k_m, sign, trnspts) =
        get_tsp_dom_params(&domain_spec, &vmax_map, &k_m_map, &sign_map, &transp_map);

    let i = trnspts
        .iter()
        .position(|d| *d != 0)
        .expect("No transporter molecule identified");

    let signed_n = sign * trnspts[i];

    let molecule = &molecules[i];
    kwargs.set_item("km", k_m).unwrap();
    kwargs.set_item("vmax", v_max).unwrap();
    kwargs.set_item("is_exporter", signed_n < 0).unwrap();
    kwargs.set_item("molecule", molecule.to_string()).unwrap();
}

// set domain specification for regulatory domain on PyDict
fn set_regulatory_domain_dict(
    kwargs: &Bound<PyDict>,
    molecules: &Vec<String>,
    domain_spec: &DomainSpecType,
    hill_map: &HashMap<u8, i8>,
    k_m_map: &HashMap<u8, f32>,
    sign_map: &HashMap<u8, i8>,
    effect_map: &HashMap<u16, Vec<i8>>,
    n_mols: &usize,
) {
    let (hill, k_m, sign, effectors) =
        get_reg_dom_params(&domain_spec, &hill_map, &k_m_map, &sign_map, &effect_map);

    let i = effectors
        .iter()
        .position(|d| *d != 0)
        .expect("No effector molecule identified");

    let signed_n = sign * effectors[i];

    let mol_i: usize;
    let is_trns: bool;
    if i < *n_mols {
        mol_i = i;
        is_trns = false;
    } else {
        mol_i = i - n_mols;
        is_trns = true;
    }

    let effector = &molecules[mol_i];
    kwargs.set_item("km", k_m).unwrap();
    kwargs.set_item("hill", hill).unwrap();
    kwargs.set_item("is_transmembrane", is_trns).unwrap();
    kwargs.set_item("is_inhibiting", signed_n < 0).unwrap();
    kwargs.set_item("effector", effector.to_string()).unwrap();
}

// Get protein representation for Protein class as PyDict
// from Protein specification using index mappings
fn get_protein_dict<'py>(
    py: Python<'py>,
    protein: &ProteinSpecType,
    molecules: &Vec<String>,
    k_m_map: &HashMap<u8, f32>,
    vmax_map: &HashMap<u8, f32>,
    sign_map: &HashMap<u8, i8>,
    hill_map: &HashMap<u8, i8>,
    reaction_map: &HashMap<u16, Vec<i8>>,
    transporter_map: &HashMap<u16, Vec<i8>>,
    effector_map: &HashMap<u16, Vec<i8>>,
    n_mols: &usize,
) -> Bound<'py, PyDict> {
    let domains: Vec<Bound<PyDict>> = (protein.0)
        .iter()
        .map(|domain_spec| {
            // idcs structure: domtype, idx0, idx1, idx2, idx3
            // where domtype: 1=catalytical, 2=transporter, 3=regulatory
            let domtype = domain_spec.0 .0;
            let kwargs = PyDict::new(py);
            kwargs.set_item("start", domain_spec.1).unwrap();
            kwargs.set_item("end", domain_spec.2).unwrap();
            if domtype == 1 {
                set_catalytic_domain_dict(
                    &kwargs,
                    molecules,
                    domain_spec,
                    vmax_map,
                    k_m_map,
                    sign_map,
                    &reaction_map,
                )
            } else if domtype == 2 {
                set_transporter_domain_dict(
                    &kwargs,
                    molecules,
                    domain_spec,
                    vmax_map,
                    k_m_map,
                    sign_map,
                    &transporter_map,
                )
            } else if domtype == 3 {
                set_regulatory_domain_dict(
                    &kwargs,
                    molecules,
                    domain_spec,
                    hill_map,
                    k_m_map,
                    sign_map,
                    effector_map,
                    n_mols,
                )
            }
            let out = PyDict::new(py);
            out.set_item("spec", kwargs).unwrap();
            out.set_item("type", get_domtype_char(&domtype)).unwrap();
            out
        })
        .collect();

    let kwargs = PyDict::new(py);
    kwargs.set_item("cds_start", protein.1).unwrap();
    kwargs.set_item("cds_end", protein.2).unwrap();
    kwargs.set_item("is_fwd", protein.3).unwrap();
    kwargs.set_item("domains", domains).unwrap();
    kwargs
}

// Get proteome representation for many Protein classes as PyDicts
// from Proteome specification using index mappings
pub fn get_proteome_dict<'py>(
    py: Python<'py>,
    proteome: &Vec<ProteinSpecType>,
    molecules: &Vec<String>,
    k_m_map: &HashMap<u8, f32>,
    vmax_map: &HashMap<u8, f32>,
    sign_map: &HashMap<u8, i8>,
    hill_map: &HashMap<u8, i8>,
    reaction_map: &HashMap<u16, Vec<i8>>,
    transporter_map: &HashMap<u16, Vec<i8>>,
    effector_map: &HashMap<u16, Vec<i8>>,
) -> Vec<Bound<'py, PyDict>> {
    let n_mols = molecules.len();
    proteome
        .iter()
        .map(|d| {
            get_protein_dict(
                py,
                &d,
                molecules,
                k_m_map,
                vmax_map,
                sign_map,
                hill_map,
                reaction_map,
                transporter_map,
                effector_map,
                &n_mols,
            )
        })
        .collect()
}
