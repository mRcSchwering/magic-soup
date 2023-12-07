use pyo3::prelude::*;
use std::collections::HashMap;

extern crate pyo3;
extern crate rand;
extern crate rand_distr;
extern crate rayon;

mod genetics;
mod mutations;

#[pyfunction]
fn get_coding_regions(
    seq: &str,
    min_cds_size: usize,
    start_codons: Vec<String>,
    stop_codons: Vec<String>,
    is_fwd: bool,
) -> Vec<(String, usize, usize, bool)> {
    let res = genetics::get_coding_regions(seq, &min_cds_size, &start_codons, &stop_codons, is_fwd);
    res
}

#[pyfunction]
fn translate_genomes(
    py: Python<'_>,
    genomes: Vec<String>,
    start_codons: Vec<String>,
    stop_codons: Vec<String>,
    domain_map: HashMap<String, usize>,
    one_codon_map: HashMap<String, usize>,
    two_codon_map: HashMap<String, usize>,
    dom_size: usize,
    dom_type_size: usize,
) -> Vec<Vec<genetics::ProteinSpecType>> {
    // TODO: always release GIL (py.allow_threads)?
    let res = py.allow_threads(|| {
        genetics::translate_genomes(
            &genomes,
            &start_codons,
            &stop_codons,
            &domain_map,
            &one_codon_map,
            &two_codon_map,
            &dom_size,
            &dom_type_size,
        )
    });
    res
}

#[pyfunction]
fn point_mutations(seqs: Vec<String>, p: f64, p_indel: f64, p_del: f64) -> Vec<(String, usize)> {
    let res = mutations::point_mutate_seqs(seqs, p, p_indel, p_del);
    res
}

#[pymodule]
fn _lib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(point_mutations, m)?)?;
    m.add_function(wrap_pyfunction!(translate_genomes, m)?)?;
    m.add_function(wrap_pyfunction!(get_coding_regions, m)?)?;
    Ok(())
}
