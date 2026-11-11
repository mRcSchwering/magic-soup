extern crate pyo3;
extern crate rand;
extern crate rand_distr;
extern crate rayon;
mod config;
mod genomics;
pub mod homology;
pub mod indel;
mod mutations;
pub mod seq;
pub mod substitutions;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use std::collections::HashMap;

// TODO: factor out all unnecessary code from lib.rs
//       e.g. into separate classes like Mutator with inner?
//       keep other modules python-free

// Mutator

#[pyclass]
pub struct Mutator {
    inner: mutations::Mutator,
}

#[pymethods]
impl Mutator {
    #[new]
    fn new(
        seed: u64,
        sub_rate: f64,
        transition_prob: f64,
        indel_rate: f64,
        indel_repeat_factor: f64,
        indel_max_multiplier: f64,
        indel_max_unit: usize,
        indel_mean_len: f64,
        indel_max_len: usize,
        insertion_prob: f64,
        break_rate: f64,
        seed_len: usize,
        min_homology: usize,
        saturating_homology: usize,
        min_segment: usize,
        duplication_prob: f64,
        min_genome_len: usize,
        max_genome_len: Option<usize>,
    ) -> PyResult<Self> {
        let cfg = config::MutatorConfig {
            sub_rate,
            transition_prob,
            indel_rate,
            indel_repeat_factor,
            indel_max_multiplier,
            indel_max_unit,
            indel_mean_len,
            indel_max_len,
            insertion_prob,
            break_rate,
            seed_len,
            min_homology,
            saturating_homology,
            min_segment,
            duplication_prob,
            min_genome_len,
            max_genome_len,
        };
        let inner = mutations::Mutator::new(cfg, seed).map_err(PyValueError::new_err)?;
        Ok(Mutator { inner })
    }

    fn mutate<'py>(
        &mut self,
        py: Python<'py>,
        buffer: PyReadonlyArray1<'py, u8>,
        starts: PyReadonlyArray1<'py, isize>,
        lens: PyReadonlyArray1<'py, isize>,
    ) -> (Bound<'py, PyArray1<i32>>, Vec<String>) {
        let slices: Vec<&[u8]> = seq::buffer_to_vec(
            buffer.as_slice().expect("buffer not contiguous"),
            starts.as_slice().expect("starts not contiguous"),
            lens.as_slice().expect("lens not contiguous"),
        );

        // mutate
        let inner = &mut self.inner;
        let changed = py.detach(|| inner.mutate(&slices));

        // return
        let idxs: Vec<i32> = changed.iter().map(|(i, _)| *i as i32).collect();
        let genomes: Vec<String> = changed
            .into_iter()
            .map(|(_, d)| String::from_utf8(d).expect("non ASCII genome"))
            .collect();

        (PyArray1::from_vec(py, idxs), genomes)
    }
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
    min_cds_size: usize,
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
        min_cds_size: usize,
    ) -> Self {
        Genomics {
            start_codons,
            stop_codons,
            domain_type_2_token,
            one_codon_2_token,
            two_codon_2_token,
            dom_size,
            dom_type_size,
            min_cds_size,
        }
    }

    fn translate_genomes<'py>(
        &self,
        py: Python<'py>,
        buffer: PyReadonlyArray1<'py, u8>,
        starts: PyReadonlyArray1<'py, isize>,
        lens: PyReadonlyArray1<'py, isize>,
    ) -> Vec<Vec<genomics::ProteinSpec>> {
        let slices: Vec<&[u8]> = seq::buffer_to_vec(
            buffer.as_slice().expect("buffer not contiguous"),
            starts.as_slice().expect("starts not contiguous"),
            lens.as_slice().expect("lens not contiguous"),
        );

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
                &self.min_cds_size,
            )
        })
    }

    fn encode_proteomes<'py>(
        &self,
        py: Python<'py>,
        buffer: PyReadonlyArray1<'py, u8>,
        starts: PyReadonlyArray1<'py, isize>,
        lens: PyReadonlyArray1<'py, isize>,
    ) -> (
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray2<i32>>,
        usize,
    ) {
        let slices: Vec<&[u8]> = seq::buffer_to_vec(
            buffer.as_slice().expect("buffer not contiguous"),
            starts.as_slice().expect("starts not contiguous"),
            lens.as_slice().expect("lens not contiguous"),
        );

        // translate genomes
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
                &self.min_cds_size,
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

// lib

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // util
    m.add_function(wrap_pyfunction!(reverse_complement, m)?)?;

    // Genomics
    m.add_class::<Genomics>()?;

    // Mutator
    m.add_class::<Mutator>()?;

    Ok(())
}
