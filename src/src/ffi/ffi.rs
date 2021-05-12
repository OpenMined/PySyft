use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn syft(_py: Python, m: &PyModule) -> PyResult<()> {
    let submod = PyModule::new(_py, "core")?;
    crate::core::core_mod(_py, submod)?;
    m.add_submodule(submod)?;
    Ok(())
}
