use pyo3::prelude::PyModule;
use pyo3::{PyResult, Python};
use serde::serde_mod_init;

pub(crate) mod serde;
pub(crate) mod uid;

pub fn common_mod_init(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<crate::core::common::uid::RustUID>()?;
    let submod = PyModule::new(_py, "serde")?;
    serde_mod_init(_py, submod)?;
    Ok(())
}
