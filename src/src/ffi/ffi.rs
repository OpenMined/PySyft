use crate::core::common::uid;
use crate::proto::core::common::Uid;
use bytes::{Bytes, BytesMut};
use pyo3::prelude::*;

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
    m.add_class::<crate::core::common::uid::RustUID>().unwrap();
    Ok(())
}
