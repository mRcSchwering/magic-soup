use pyo3::prelude::*;

extern crate pyo3;
extern crate rand;
extern crate rand_distr;
extern crate rayon;

mod mutations;


#[pyfunction]
fn point_mutations(seqs: Vec<String>, p: f64, p_indel: f64) -> Vec<(String, usize)> {
    let p_del = 0.6;
    let res = mutations::point_mutate_seqs(seqs, p, p_indel, p_del);
    res
}


#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pymodule]
fn _lib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(point_mutations, m)?)?;
    Ok(())
}