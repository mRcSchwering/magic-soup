use rayon::prelude::*;

pub type DomainSpecType = ((u8, u8, u8, u8, u16), usize, usize);
pub type ProteinSpecType = (Vec<DomainSpecType>, usize, usize, bool);

fn fmt_int_kwarg(key: &str, val: usize) -> String {
    format!("\"{key}\":{val}")
}

fn fmt_float_kwarg(key: &str, val: f32) -> String {
    format!("\"{key}\":{:.5}", val)
}

fn fmt_bool_kwarg(key: &str, val: bool) -> String {
    format!("\"{key}\":{val}")
}

fn fmt_arr_kwarg(key: &str, val: Vec<String>) -> String {
    format!("\"{key}\":[{}]", val.join(","))
}

fn push_catalytic_domain(
    kwargs: &mut Vec<String>,
    km: &f32,
    vmax: &f32,
    sign: &i8,
    react: &Vec<i8>,
    molecules: &Vec<String>,
) {
    let mut lfts: Vec<String> = vec![];
    let mut rgts: Vec<String> = vec![];
    for (mol_i, n) in react.iter().enumerate() {
        let signed_n = *n * sign;
        if signed_n == 0 {
            continue;
        } else if signed_n > 0 {
            let mol = &molecules[mol_i];
            rgts.extend((0..n.abs()).map(|_| format!("\"{}\"", mol.to_string())));
        } else {
            let mol = &molecules[mol_i];
            lfts.extend((0..n.abs()).map(|_| format!("\"{}\"", mol.to_string())));
        }
    }
    kwargs.extend([
        fmt_float_kwarg("km", *km),
        fmt_float_kwarg("vmax", *vmax),
        format!("\"reaction\":[[{}],[{}]]", lfts.join(","), rgts.join(",")),
    ])
}

fn push_transporter_domain(
    kwargs: &mut Vec<String>,
    km: &f32,
    vmax: &f32,
    sign: &i8,
    trnspts: &Vec<i8>,
    molecules: &Vec<String>,
) {
    let mol_i = trnspts.iter().position(|d| *d != 0).unwrap();
    let signed_n = trnspts[mol_i] * sign;
    kwargs.extend([
        fmt_float_kwarg("km", *km),
        fmt_float_kwarg("vmax", *vmax),
        fmt_bool_kwarg("is_exporter", signed_n < 0),
        format!("\"molecule\":\"{}\"", molecules[mol_i]),
    ])
}

fn push_regulatory_domain(
    kwargs: &mut Vec<String>,
    km: &f32,
    hill: &u8,
    sign: &i8,
    effectors: &Vec<i8>,
    molecules: &Vec<String>,
    n_mols: &usize,
) {
    let i = effectors.iter().position(|d| *d != 0).unwrap();
    let signed_n = effectors[i] * sign;
    let mol_i: usize;
    let is_transmembrane: bool;
    if i < *n_mols {
        mol_i = i;
        is_transmembrane = false;
    } else {
        mol_i = i - n_mols;
        is_transmembrane = true;
    }
    kwargs.extend([
        fmt_float_kwarg("km", *km),
        fmt_int_kwarg("hill", *hill as usize),
        fmt_bool_kwarg("is_transmembrane", is_transmembrane),
        fmt_bool_kwarg("is_inhibiting", signed_n < 0),
        format!("\"effector\":\"{}\"", molecules[mol_i]),
    ])
}

// Get protein init kwargs as JSON parsable string
// yes this is ridiculous
fn get_protein(
    protein: &ProteinSpecType,
    vmaxs: &Vec<f32>,
    kms: &Vec<f32>,
    hills: &Vec<u8>,
    signs: &Vec<i8>,
    reacts: &Vec<Vec<i8>>,
    trnspts: &Vec<Vec<i8>>,
    effectors: &Vec<Vec<i8>>,
    molecules: &Vec<String>,
    n_mols: &usize,
) -> String {
    let domstrs: Vec<String> = (protein.0)
        .iter()
        .enumerate()
        .map(|(dom_i, (idxs, start, end))| {
            let domtype = idxs.0;
            let mut kwargs = vec![fmt_int_kwarg("start", *start), fmt_int_kwarg("end", *end)];
            if domtype == 1 {
                push_catalytic_domain(
                    &mut kwargs,
                    &kms[dom_i],
                    &vmaxs[dom_i],
                    &signs[dom_i],
                    &reacts[dom_i],
                    molecules,
                )
            } else if domtype == 2 {
                push_transporter_domain(
                    &mut kwargs,
                    &kms[dom_i],
                    &vmaxs[dom_i],
                    &signs[dom_i],
                    &trnspts[dom_i],
                    molecules,
                )
            } else if domtype == 3 {
                push_regulatory_domain(
                    &mut kwargs,
                    &kms[dom_i],
                    &hills[dom_i],
                    &signs[dom_i],
                    &effectors[dom_i],
                    molecules,
                    n_mols,
                )
            }
            format!("[{},{{{}}}]", domtype, kwargs.join(","))
        })
        .collect();

    let kwargs = [
        fmt_int_kwarg("cds_start", protein.1),
        fmt_int_kwarg("cds_end", protein.2),
        fmt_bool_kwarg("is_fwd", protein.3),
        fmt_arr_kwarg("domains", domstrs),
    ];

    format!("{{{}}}", kwargs.join(","))
}

// Threaded version of get_protein() for multiple proteomes
// returned in a JSON parsable string in {"data": [...]}
pub fn get_proteome_threaded(
    proteome: &Vec<ProteinSpecType>,
    vmaxs: &Vec<Vec<f32>>,
    kms: &Vec<Vec<f32>>,
    hills: &Vec<Vec<u8>>,
    signs: &Vec<Vec<i8>>,
    reacts: &Vec<Vec<Vec<i8>>>,
    trnspts: &Vec<Vec<Vec<i8>>>,
    effectors: &Vec<Vec<Vec<i8>>>,
    molecules: &Vec<String>,
) -> String {
    let n_mols = molecules.len();
    let proteins: Vec<String> = proteome
        .into_par_iter()
        .enumerate()
        .map(|(i, d)| {
            get_protein(
                &d,
                &vmaxs[i],
                &kms[i],
                &hills[i],
                &signs[i],
                &reacts[i],
                &trnspts[i],
                &effectors[i],
                molecules,
                &n_mols,
            )
        })
        .collect();

    format!("{{\"data\":[{}]}}", proteins.join(","))
}
