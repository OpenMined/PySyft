use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn hello_rust() -> PyResult<String> {
    Ok("Hello Rust 🦀".to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn libsyft(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(hello_rust))?;

    Ok(())
}
