extern crate pyo3;
extern crate rand;
extern crate rand_distr;
extern crate rayon;

mod genetics;
mod mutations;

use pyo3::prelude::*;
use std::collections::HashMap;

// mutations

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

#[pyfunction]
fn recombinations(
    py: Python<'_>,
    seq_pairs: Vec<(String, String)>,
    p: f64,
) -> Vec<(String, String, usize)> {
    py.allow_threads(move || mutations::recombinate_seq_pairs(seq_pairs, p))
}

// genetics

#[pyfunction]
fn get_coding_regions(
    seq: &str,
    min_cds_size: usize,
    start_codons: Vec<String>,
    stop_codons: Vec<String>,
    is_fwd: bool,
) -> Vec<(usize, usize, bool)> {
    genetics::get_coding_regions(seq, &min_cds_size, &start_codons, &stop_codons, is_fwd)
}

#[pyfunction]
fn extract_domains(
    genome: String,
    cdss: Vec<(usize, usize, bool)>,
    dom_size: usize,
    dom_type_size: usize,
    dom_type_map: HashMap<String, usize>,
    one_codon_map: HashMap<String, usize>,
    two_codon_map: HashMap<String, usize>,
) -> Vec<genetics::ProteinSpecType> {
    genetics::extract_domains(
        &genome,
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
    py.allow_threads(move || {
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

// lib

#[pymodule]
fn _lib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // mutations
    m.add_function(wrap_pyfunction!(point_mutations, m)?)?;
    m.add_function(wrap_pyfunction!(recombinations, m)?)?;

    // genetics
    m.add_function(wrap_pyfunction!(get_coding_regions, m)?)?;
    m.add_function(wrap_pyfunction!(extract_domains, m)?)?;
    m.add_function(wrap_pyfunction!(reverse_complement, m)?)?;
    m.add_function(wrap_pyfunction!(translate_genomes, m)?)?;
    Ok(())
}
