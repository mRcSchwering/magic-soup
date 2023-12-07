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
    genetics::get_coding_regions(seq, &min_cds_size, &start_codons, &stop_codons, is_fwd)
}

#[pyfunction]
fn extract_domains(
    cdss: Vec<(String, usize, usize, bool)>,
    dom_size: usize,
    dom_type_size: usize,
    dom_type_map: HashMap<String, usize>,
    one_codon_map: HashMap<String, usize>,
    two_codon_map: HashMap<String, usize>,
) -> Vec<genetics::ProteinSpecType> {
    genetics::extract_domains(
        &cdss,
        &dom_size,
        &dom_type_size,
        &dom_type_map,
        &one_codon_map,
        &two_codon_map,
    )
}

#[pyfunction]
fn reverse_complement(seq: String) -> String {
    genetics::reverse_complement(&seq)
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
    py.allow_threads(|| {
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
    })
}

#[pyfunction]
fn point_mutations(
    py: Python<'_>,
    seqs: Vec<String>,
    p: f64,
    p_indel: f64,
    p_del: f64,
) -> Vec<(String, usize)> {
    py.allow_threads(move || mutations::point_mutate_seqs(seqs, p, p_indel, p_del))
}

#[pymodule]
fn _lib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(point_mutations, m)?)?;
    m.add_function(wrap_pyfunction!(translate_genomes, m)?)?;
    m.add_function(wrap_pyfunction!(get_coding_regions, m)?)?;
    m.add_function(wrap_pyfunction!(extract_domains, m)?)?;
    m.add_function(wrap_pyfunction!(reverse_complement, m)?)?;
    Ok(())
}
