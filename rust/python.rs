//! PyO3 bindings.
//!
//! Python usage:
//!     m = genome_mutator.Mutator(seed=1, sub_rate=1e-5, break_rate=1e-4)
//!     for idx, new_seq in m.mutate(genomes):   # genomes: list[str]
//!         genomes[idx] = new_seq
//!     print(m.stats())

use crate::config::Config;
use crate::engine::Mutator;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyDict;

#[pyclass(name = "Mutator")]
pub struct PyMutator {
    inner: Mutator,
}

#[pymethods]
impl PyMutator {
    #[new]
    #[pyo3(signature = (
        seed = 0,
        sub_rate = 1e-5,
        transition_prob = 0.55,
        indel_rate = 1e-7,
        indel_repeat_factor = 2.0,
        indel_max_multiplier = 64.0,
        indel_max_unit = 6,
        indel_mean_len = 1.5,
        indel_max_len = 20,
        insertion_prob = 0.5,
        break_rate = 1e-4,
        seed_len = 12,
        min_homology = 15,
        saturating_homology = 60,
        min_segment = 20,
        duplication_prob = 0.5,
        min_genome_len = 50,
        max_genome_len = None,
    ))]
    #[allow(clippy::too_many_arguments)]
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
        let cfg = Config {
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
        let inner = Mutator::new(cfg, seed).map_err(PyValueError::new_err)?;
        Ok(PyMutator { inner })
    }

    /// Mutate one generation. `genomes` is a list of ASCII str (A/C/G/T).
    /// Returns [(index, new_sequence), ...] for genomes that changed only.
    fn mutate(
        &mut self,
        py: Python<'_>,
        genomes: Vec<PyBackedStr>,
    ) -> PyResult<Vec<(usize, String)>> {
        // PyBackedStr borrows the Python string's UTF-8 buffer without copying
        let slices: Vec<&[u8]> = genomes.iter().map(|g| g.as_bytes()).collect();
        let inner = &mut self.inner;
        let changed = py.allow_threads(|| inner.mutate(&slices));
        changed
            .into_iter()
            .map(|(i, v)| {
                String::from_utf8(v)
                    .map(|s| (i, s))
                    .map_err(|_| PyValueError::new_err(format!("genome {i} is not ASCII")))
            })
            .collect()
    }

    /// Cumulative event counts since creation or the last reset_stats().
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = &self.inner.stats;
        let d = PyDict::new_bound(py);
        d.set_item("substitutions", s.substitutions)?;
        d.set_item("indel_candidates", s.indel_candidates)?;
        d.set_item("insertions", s.insertions)?;
        d.set_item("deletions", s.deletions)?;
        d.set_item("repeat_indels", s.repeat_indels)?;
        d.set_item("breaks", s.breaks)?;
        d.set_item("breaks_no_partner", s.breaks_no_partner)?;
        d.set_item("breaks_rejected", s.breaks_rejected)?;
        d.set_item("large_deletions", s.large_deletions)?;
        d.set_item("large_duplications", s.large_duplications)?;
        d.set_item("inversions", s.inversions)?;
        Ok(d)
    }

    fn reset_stats(&mut self) {
        self.inner.stats = Default::default();
    }
}

#[pymodule]
fn genome_mutator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMutator>()?;
    Ok(())
}
