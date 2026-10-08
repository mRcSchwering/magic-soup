extern crate pyo3;
extern crate rand;
extern crate rand_distr;
extern crate rayon;

mod genetics;
mod genomics;
mod mutations;
mod proteomics;
mod util;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

// util

#[pyfunction]
fn dist_1d(a: u16, b: u16, m: u16) -> u16 {
    util::dist_1d(&a, &b, &m)
}

#[pyfunction]
fn reverse_complement(seq: String) -> String {
    util::reverse_complement(&seq)
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

// Genomics

#[pyclass]
struct Genomics {
    start_codons: Vec<String>,
    stop_codons: Vec<String>,
    domain_map: HashMap<String, u8>,
    one_codon_map: HashMap<String, u8>,
    two_codon_map: HashMap<String, u16>,
    dom_size: u8,
    dom_type_size: u8,
}

#[pymethods]
impl Genomics {
    #[new]
    fn new(
        start_codons: Vec<String>,
        stop_codons: Vec<String>,
        domain_map: HashMap<String, u8>,
        one_codon_map: HashMap<String, u8>,
        two_codon_map: HashMap<String, u16>,
        dom_size: u8,
        dom_type_size: u8,
    ) -> Self {
        Genomics {
            start_codons,
            stop_codons,
            domain_map,
            one_codon_map,
            two_codon_map,
            dom_size,
            dom_type_size,
        }
    }

    fn translate_genomes(
        &self,
        py: Python<'_>,
        genomes: Vec<String>,
    ) -> Vec<Vec<genomics::ProteinSpecType>> {
        py.allow_threads(move || {
            genomics::translate_genomes_threaded(
                &genomes,
                &self.start_codons,
                &self.stop_codons,
                &self.domain_map,
                &self.one_codon_map,
                &self.two_codon_map,
                &self.dom_size,
                &self.dom_type_size,
            )
        })
    }

    fn get_coding_regions(
        &self,
        seq: &str,
        min_cds_size: u8,
        is_fwd: bool,
    ) -> Vec<(usize, usize, bool)> {
        genomics::get_coding_regions(
            seq,
            &min_cds_size,
            &self.start_codons,
            &self.stop_codons,
            is_fwd,
        )
    }

    fn extract_domains(
        &self,
        genome: String,
        cdss: Vec<(usize, usize, bool)>,
    ) -> Vec<genetics::ProteinSpecType> {
        genomics::extract_domains(
            &genome,
            &cdss,
            &self.dom_size,
            &self.dom_type_size,
            &self.domain_map,
            &self.one_codon_map,
            &self.two_codon_map,
        )
    }
}

// Proteomics

#[pyclass]
struct Proteomics {
    k_m_map: HashMap<u8, f32>,
    vmax_map: HashMap<u8, f32>,
    sign_map: HashMap<u8, i8>,
    hill_map: HashMap<u8, i8>,
    reaction_map: HashMap<u16, Vec<i8>>,
    transporter_map: HashMap<u16, Vec<i8>>,
    effector_map: HashMap<u16, Vec<i8>>,
    m: usize,
}

#[pymethods]
impl Proteomics {
    #[new]
    fn new(
        k_m_map: HashMap<u8, f32>,
        vmax_map: HashMap<u8, f32>,
        sign_map: HashMap<u8, i8>,
        hill_map: HashMap<u8, i8>,
        reaction_map: HashMap<u16, Vec<i8>>,
        transporter_map: HashMap<u16, Vec<i8>>,
        effector_map: HashMap<u16, Vec<i8>>,
        m: usize,
    ) -> Self {
        Proteomics {
            k_m_map,
            vmax_map,
            sign_map,
            hill_map,
            reaction_map,
            transporter_map,
            effector_map,
            m,
        }
    }

    fn get_proteome_params(
        &self,
        py: Python<'_>,
        proteomes: Vec<Vec<genomics::ProteinSpecType>>,
    ) -> (
        util::MatrixType<f32>,
        util::MatrixType<f32>,
        util::MatrixType<Vec<f32>>,
        util::MatrixType<Vec<i8>>,
        util::MatrixType<Vec<i8>>,
        util::MatrixType<Vec<i8>>,
    ) {
        py.allow_threads(move || {
            proteomics::get_proteome_params_threaded(
                &proteomes,
                &self.k_m_map,
                &self.vmax_map,
                &self.sign_map,
                &self.hill_map,
                &self.reaction_map,
                &self.transporter_map,
                &self.effector_map,
                &self.m,
            )
        })
    }

    fn get_proteome_dict<'py>(
        &self,
        py: Python<'py>,
        proteome: Vec<genomics::ProteinSpecType>,
        molecules: Vec<String>,
    ) -> Vec<Bound<'py, PyDict>> {
        proteomics::get_proteome_dict(
            py,
            &proteome,
            &molecules,
            &self.k_m_map,
            &self.vmax_map,
            &self.sign_map,
            &self.hill_map,
            &self.reaction_map,
            &self.transporter_map,
            &self.effector_map,
        )
    }
}

// Cells

#[pyclass]
struct Cells {
    genomics: Py<Genomics>,
    proteomics: Py<Proteomics>,
}

#[pymethods]
impl Cells {
    #[new]
    fn new(genomics: Bound<Genomics>, proteomics: Bound<Proteomics>) -> Self {
        Cells {
            genomics: genomics.unbind(),
            proteomics: proteomics.unbind(),
        }
    }

    fn translate_genomes(
        &self,
        py: Python,
        genomes: Vec<String>,
    ) -> (
        util::MatrixType<f32>,
        util::MatrixType<f32>,
        util::MatrixType<Vec<f32>>,
        util::MatrixType<Vec<i8>>,
        util::MatrixType<Vec<i8>>,
        util::MatrixType<Vec<i8>>,
    ) {
        // idiomatic way of using PyO3 class instance reference
        let genomics = self.genomics.borrow(py);
        let proteomics = self.proteomics.borrow(py);

        let proteomes = genomics.translate_genomes(py, genomes);
        proteomics.get_proteome_params(py, proteomes)
    }
}

// lib

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // util
    m.add_function(wrap_pyfunction!(dist_1d, m)?)?;
    m.add_function(wrap_pyfunction!(reverse_complement, m)?)?;

    // mutations
    m.add_function(wrap_pyfunction!(point_mutations, m)?)?;
    m.add_function(wrap_pyfunction!(recombinations, m)?)?;

    // Genomics
    m.add_class::<Genomics>()?;

    //Proteomics
    m.add_class::<Proteomics>()?;

    // Cells
    m.add_class::<Cells>()?;

    Ok(())
}
