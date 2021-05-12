use crate::core::common::serde::serializable::Serializable;
use crate::core::common::uid::RustUID;
use bytes::BytesMut;
use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule};

#[pyfunction]
fn uid_serde(obj: &PyAny) -> PyResult<Vec<u8>> {
    let uid_proto: Uid = obj.extract().unwrap();
    let buf: Vec<u8> = uid_proto.serialize();
    Ok(buf)
    //bug here
}

/// A Python module implemented in Rust.
#[pymodule]
fn syft(_py: Python, m: &PyModule) -> PyResult<()> {
    let submod = PyModule::new(_py, "core")?;
    crate::core::core_mod(_py, submod);
    m.add_submodule(submod);
    Ok(())
}
