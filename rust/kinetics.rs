use pyo3::marker::Python;
use pyo3::types::PyDict;

pub type DomainSpecType = ((u8, u8, u8, u8, u16), usize, usize);
pub type ProteinSpecType = (Vec<DomainSpecType>, usize, usize, bool);

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
fn set_catalytic_domain(
    kwargs: &PyDict,
    km: &f32,
    vmax: &f32,
    sign: &i8,
    react: &Vec<i8>,
    molecules: &Vec<String>,
) {
    // must be at least 1 for each
    let mut lfts: Vec<String> = Vec::with_capacity(2);
    let mut rgts: Vec<String> = Vec::with_capacity(2);
    for (mol_i, n) in react.iter().enumerate() {
        let signed_n = *n * sign;
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
    kwargs.set_item("km", km).unwrap();
    kwargs.set_item("vmax", vmax).unwrap();
    kwargs.set_item("reaction", (lfts, rgts)).unwrap();
}

// set domain specification for transporter domain on PyDict
fn set_transporter_domain(
    kwargs: &PyDict,
    km: &f32,
    vmax: &f32,
    sign: &i8,
    trnspts: &Vec<i8>,
    molecules: &Vec<String>,
) {
    let mol_i = trnspts
        .iter()
        .position(|d| *d != 0)
        .expect("No transporter molecule identified");
    let signed_n = trnspts[mol_i] * sign;
    let molecule = &molecules[mol_i];
    kwargs.set_item("km", km).unwrap();
    kwargs.set_item("vmax", vmax).unwrap();
    kwargs.set_item("is_exporter", signed_n < 0).unwrap();
    kwargs.set_item("molecule", molecule.to_string()).unwrap();
}

// set domain specification for regulatory domain on PyDict
fn set_regulatory_domain(
    kwargs: &PyDict,
    km: &f32,
    hill: &u8,
    sign: &i8,
    effectors: &Vec<i8>,
    molecules: &Vec<String>,
    n_mols: &usize,
) {
    let i = effectors
        .iter()
        .position(|d| *d != 0)
        .expect("No effector molecule identified");
    let signed_n = effectors[i] * sign;
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
    kwargs.set_item("km", km).unwrap();
    kwargs.set_item("hill", hill).unwrap();
    kwargs.set_item("is_transmembrane", is_trns).unwrap();
    kwargs.set_item("is_inhibiting", signed_n < 0).unwrap();
    kwargs.set_item("effector", effector.to_string()).unwrap();
}

// Get protein specification for Protein class as PyDict from
// indexes Protein specification using index mappings
fn get_protein<'py>(
    py: Python<'py>,
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
) -> &'py PyDict {
    let domains: Vec<&PyDict> = (protein.0)
        .iter()
        .enumerate()
        .map(|(dom_i, (idxs, start, end))| {
            // idcs structure: domtype, idx0, idx1, idx2, idx3
            // where domtype: 1=catalytical, 2=transporter, 3=regulatory
            let domtype = idxs.0;
            let kwargs = PyDict::new(py);
            kwargs.set_item("start", start).unwrap();
            kwargs.set_item("end", end).unwrap();
            if domtype == 1 {
                set_catalytic_domain(
                    kwargs,
                    &kms[dom_i],
                    &vmaxs[dom_i],
                    &signs[dom_i],
                    &reacts[dom_i],
                    molecules,
                )
            } else if domtype == 2 {
                set_transporter_domain(
                    kwargs,
                    &kms[dom_i],
                    &vmaxs[dom_i],
                    &signs[dom_i],
                    &trnspts[dom_i],
                    molecules,
                )
            } else if domtype == 3 {
                set_regulatory_domain(
                    kwargs,
                    &kms[dom_i],
                    &hills[dom_i],
                    &signs[dom_i],
                    &effectors[dom_i],
                    molecules,
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

// Get proteome specification for many Protein classes as PyDicts from
// indexes Proteome specification using index mappings
pub fn get_proteome<'py>(
    py: Python<'py>,
    proteome: &Vec<ProteinSpecType>,
    vmaxs: &Vec<Vec<f32>>,
    kms: &Vec<Vec<f32>>,
    hills: &Vec<Vec<u8>>,
    signs: &Vec<Vec<i8>>,
    reacts: &Vec<Vec<Vec<i8>>>,
    trnspts: &Vec<Vec<Vec<i8>>>,
    effectors: &Vec<Vec<Vec<i8>>>,
    molecules: &Vec<String>,
) -> Vec<&'py PyDict> {
    let n_mols = molecules.len();
    proteome
        .iter()
        .enumerate()
        .map(|(i, d)| {
            get_protein(
                py,
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
        .collect()
}
