use common::common_mod_init;
use pyo3::prelude::PyModule;
use pyo3::{PyResult, Python};

pub(crate) mod common;
pub(crate) mod store;

pub fn core_mod(py: Python, module: &PyModule) -> PyResult<()> {
    let submod = PyModule::new(py, "common")?;
    common_mod_init(py, submod)?;
    module.add_submodule(submod)?;
    Ok(())
}
