use crate::core::common::uid;
use crate::proto::core::common::Uid;
use bytes::{Bytes, BytesMut};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::wrap_pyfunction;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn hello_rust() -> PyResult<String> {
    Ok("Hello Rust 🦀".to_string())
}

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
    m.add_wrapped(wrap_pyfunction!(hello_rust))?;
    m.add_wrapped(wrap_pyfunction!(uid_serde))?;
    m.add_class::<uid::RustUID>();
    Ok(())
}
