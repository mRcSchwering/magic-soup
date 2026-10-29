extern crate pyo3;
extern crate rand;
extern crate rand_distr;
extern crate rayon;
mod genomics;
mod mutations;
mod proteomics;
mod util;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyDict;
use std::collections::HashMap;

// mutations

#[pyfunction]
fn point_mutations(
    py: Python<'_>,
    seqs: Vec<String>,
    p: f32,
    p_indel: f32,
    p_del: f32,
) -> Vec<(String, usize)> {
    py.detach(move || mutations::point_mutations_threaded(seqs, p, p_indel, p_del))
}

#[pyfunction]
fn recombinations(
    py: Python<'_>,
    seq_pairs: Vec<(String, String)>,
    p: f32,
) -> Vec<(String, String, usize)> {
    py.detach(move || mutations::recombinations_threaded(seq_pairs, p))
}

// Genomics

#[pyfunction]
fn reverse_complement(seq: PyBackedStr) -> String {
    let bytes = genomics::reverse_complement(&seq.as_bytes());
    String::from_utf8(bytes).expect("Genome not ASCII")
}

#[pyclass]
struct Genomics {
    start_codons: Vec<[u8; genomics::CODON_SIZE]>,
    stop_codons: Vec<[u8; genomics::CODON_SIZE]>,
    domain_type_2_token: HashMap<[u8; 2 * genomics::CODON_SIZE], u8>,
    one_codon_2_token: HashMap<[u8; genomics::CODON_SIZE], u8>,
    two_codon_2_token: HashMap<[u8; 2 * genomics::CODON_SIZE], u16>,
    dom_size: usize,
    dom_type_size: usize,
}

#[pymethods]
impl Genomics {
    #[new]
    fn new(
        start_codons: Vec<[u8; genomics::CODON_SIZE]>,
        stop_codons: Vec<[u8; genomics::CODON_SIZE]>,
        domain_type_2_token: HashMap<[u8; 2 * genomics::CODON_SIZE], u8>,
        one_codon_2_token: HashMap<[u8; genomics::CODON_SIZE], u8>,
        two_codon_2_token: HashMap<[u8; 2 * genomics::CODON_SIZE], u16>,
        dom_size: usize,
        dom_type_size: usize,
    ) -> Self {
        Genomics {
            start_codons,
            stop_codons,
            domain_type_2_token,
            one_codon_2_token,
            two_codon_2_token,
            dom_size,
            dom_type_size,
        }
    }

    fn translate_genomes(
        &self,
        py: Python<'_>,
        genomes: Vec<PyBackedStr>,
    ) -> Vec<Vec<genomics::ProteinSpec>> {
        let slices: Vec<&[u8]> = genomes.iter().map(|g| g.as_bytes()).collect();
        py.detach(move || {
            genomics::translate_genomes_threaded(
                &slices,
                &self.start_codons,
                &self.stop_codons,
                &self.domain_type_2_token,
                &self.one_codon_2_token,
                &self.two_codon_2_token,
                &self.dom_size,
                &self.dom_type_size,
            )
        })
    }

    fn translate_genomes_new<'py>(
        &self,
        py: Python<'py>,
        genomes: Vec<PyBackedStr>,
    ) -> (
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray2<i32>>,
        usize,
    ) {
        // translate genomes
        let slices: Vec<&[u8]> = genomes.iter().map(|g| g.as_bytes()).collect();
        let specs = py.detach(move || {
            genomics::translate_genomes_threaded(
                &slices,
                &self.start_codons,
                &self.stop_codons,
                &self.domain_type_2_token,
                &self.one_codon_2_token,
                &self.two_codon_2_token,
                &self.dom_size,
                &self.dom_type_size,
            )
        });

        // convert to flat representation
        let mut cell: Vec<i32> = Vec::new();
        let mut prot: Vec<i32> = Vec::new();
        let mut tokens: Vec<Vec<i32>> = Vec::new();
        let mut max_p: usize = 0;

        if let Some(p) = specs.iter().map(|d| d.len()).max() {
            max_p = p;
            let n_doms: usize = specs.iter().flatten().map(|d| d.0.len()).sum();
            cell.reserve(n_doms);
            prot.reserve(n_doms);
            tokens.reserve(n_doms);

            for (cell_i, cell_spec) in specs.iter().enumerate() {
                for (prot_i, prot_spec) in cell_spec.iter().enumerate() {
                    for dom_spec in prot_spec.0.iter() {
                        cell.push(cell_i as i32);
                        prot.push(prot_i as i32);
                        let t = dom_spec.0;
                        tokens.push(vec![
                            t.0 as i32, t.1 as i32, t.2 as i32, t.3 as i32, t.4 as i32,
                        ]);
                    }
                }
            }
        }

        // fill up numpy arrays
        let cell_arr: Bound<'py, PyArray1<i32>> = PyArray1::from_vec(py, cell);
        let prot_arr: Bound<'py, PyArray1<i32>> = PyArray1::from_vec(py, prot);
        let tokens_arr: Bound<'py, PyArray2<i32>> =
            PyArray2::from_vec2(py, &tokens).expect("Failed to build PyArray2");

        (cell_arr, prot_arr, tokens_arr, max_p)
    }

    fn get_coding_regions(
        &self,
        seq: PyBackedStr,
        min_cds_size: usize,
        is_fwd: bool,
    ) -> Vec<(usize, usize, bool)> {
        genomics::get_coding_regions(
            seq.as_bytes(),
            &min_cds_size,
            &self.start_codons,
            &self.stop_codons,
            is_fwd,
        )
    }

    fn extract_domains(
        &self,
        genome: PyBackedStr,
        cdss: Vec<(usize, usize, bool)>,
    ) -> Vec<genomics::ProteinSpec> {
        genomics::extract_domains(
            &genome.as_bytes(),
            &cdss,
            &self.dom_size,
            &self.dom_type_size,
            &self.domain_type_2_token,
            &self.one_codon_2_token,
            &self.two_codon_2_token,
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
        proteomes: Vec<Vec<genomics::ProteinSpec>>,
    ) -> (
        util::MatrixType<f32>,
        util::MatrixType<f32>,
        util::MatrixType<Vec<f32>>,
        util::MatrixType<Vec<i8>>,
        util::MatrixType<Vec<i8>>,
        util::MatrixType<Vec<i8>>,
    ) {
        py.detach(move || {
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
        proteome: Vec<genomics::ProteinSpec>,
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
        genomes: Vec<PyBackedStr>,
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
