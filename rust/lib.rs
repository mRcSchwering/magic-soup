extern crate pyo3;
extern crate rand;
extern crate rand_distr;
extern crate rayon;

mod genetics;
mod kinetics;
mod mutations;
mod util;
mod world;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

// util

#[pyfunction]
fn dist_1d(a: u16, b: u16, m: u16) -> u16 {
    util::dist_1d(&a, &b, &m)
}

#[pyfunction]
fn free_moores_nghbhd(
    x: u16,
    y: u16,
    positions: Vec<(u16, u16)>,
    map_size: u16,
) -> Vec<(u16, u16)> {
    util::free_moores_nghbhd(&x, &y, &positions, &map_size)
}

// mutations

#[pyfunction]
fn point_mutations(
    py: Python<'_>,
    seqs: Vec<String>,
    p: f32,
    p_indel: f32,
    p_del: f32,
) -> Vec<(String, usize)> {
    py.allow_threads(move || mutations::point_mutations_threaded(seqs, p, p_indel, p_del))
}

#[pyfunction]
fn recombinations(
    py: Python<'_>,
    seq_pairs: Vec<(String, String)>,
    p: f32,
) -> Vec<(String, String, usize)> {
    py.allow_threads(move || mutations::recombinations_threaded(seq_pairs, p))
}

// genetics

#[pyfunction]
fn get_coding_regions(
    seq: &str,
    min_cds_size: u8,
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
    dom_size: u8,
    dom_type_size: u8,
    dom_type_map: HashMap<String, u8>,
    one_codon_map: HashMap<String, u8>,
    two_codon_map: HashMap<String, u16>,
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
    domain_map: HashMap<String, u8>,
    one_codon_map: HashMap<String, u8>,
    two_codon_map: HashMap<String, u16>,
    dom_size: u8,
    dom_type_size: u8,
) -> Vec<Vec<genetics::ProteinSpecType>> {
    py.allow_threads(move || {
        genetics::translate_genomes_threaded(
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

// world

#[pyfunction]
fn get_neighbors(
    py: Python<'_>,
    from_idxs: Vec<usize>,
    to_idxs: Vec<usize>,
    positions: Vec<(u16, u16)>,
    map_size: u16,
) -> Vec<(usize, usize)> {
    py.allow_threads(move || {
        world::get_neighbors_threaded(&from_idxs, &to_idxs, &positions, &map_size)
    })
}

#[pyfunction]
fn divide_cells_if_possible(
    py: Python<'_>,
    cell_idxs: Vec<usize>,
    positions: Vec<(u16, u16)>,
    n_cells: usize,
    map_size: u16,
) -> (Vec<usize>, Vec<usize>, Vec<(u16, u16)>) {
    py.allow_threads(move || {
        world::divide_cells_if_possible_threaded(&cell_idxs, &positions, &n_cells, &map_size)
    })
}

#[pyfunction]
fn move_cells(
    py: Python<'_>,
    cell_idxs: Vec<usize>,
    positions: Vec<(u16, u16)>,
    map_size: u16,
) -> (Vec<(u16, u16)>, Vec<usize>) {
    py.allow_threads(move || world::move_cells_threaded(&cell_idxs, &positions, &map_size))
}

// kinetics

#[pyfunction]
fn get_proteome(
    py: Python<'_>,
    proteome: Vec<kinetics::ProteinSpecType>,
    vmaxs: Vec<Vec<f32>>,
    kms: Vec<Vec<f32>>,
    hills: Vec<Vec<u8>>,
    signs: Vec<Vec<i8>>,
    reacts: Vec<Vec<Vec<i8>>>,
    trnspts: Vec<Vec<Vec<i8>>>,
    effectors: Vec<Vec<Vec<i8>>>,
    molecules: Vec<String>,
) -> Vec<&PyDict> {
    kinetics::get_proteome(
        py, &proteome, &vmaxs, &kms, &hills, &signs, &reacts, &trnspts, &effectors, &molecules,
    )
}

// lib

#[pymodule]
fn _lib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // util
    m.add_function(wrap_pyfunction!(dist_1d, m)?)?;
    m.add_function(wrap_pyfunction!(free_moores_nghbhd, m)?)?;

    // mutations
    m.add_function(wrap_pyfunction!(point_mutations, m)?)?;
    m.add_function(wrap_pyfunction!(recombinations, m)?)?;

    // genetics
    m.add_function(wrap_pyfunction!(get_coding_regions, m)?)?;
    m.add_function(wrap_pyfunction!(extract_domains, m)?)?;
    m.add_function(wrap_pyfunction!(reverse_complement, m)?)?;
    m.add_function(wrap_pyfunction!(translate_genomes, m)?)?;

    // world
    m.add_function(wrap_pyfunction!(get_neighbors, m)?)?;
    m.add_function(wrap_pyfunction!(divide_cells_if_possible, m)?)?;
    m.add_function(wrap_pyfunction!(move_cells, m)?)?;

    // kinetics
    m.add_function(wrap_pyfunction!(get_proteome, m)?)?;

    Ok(())
}
