use crate::core::common::serde::serializable::Serializable;
use crate::core::common::uid::RustUID;
use bytes::BytesMut;
use pyo3::prelude::PyModule;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub(crate) mod deserialize;
pub(crate) mod serializable;
pub(crate) mod serialize;

#[pyfunction]
pub fn py_deserialize(obj: Vec<u8>, type_name: String) -> RustUID {
    if type_name.contains("RustUID") {
        let obj = RustUID::_deserialize(BytesMut::from(obj.as_slice()));
        obj
    } else {
        unimplemented!("Not implemented");
    }
}

#[pyfunction]
pub fn py_serialize(obj: &PyAny, type_name: String) -> Vec<u8> {
    if type_name.contains("RustUID") {
        let rust_obj: RustUID = obj.extract().unwrap();
        rust_obj.serialize(false)
    } else {
        unimplemented!("Not implemented");
    }
}

pub fn serde_mod_init(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_serialize))?;
    m.add_wrapped(wrap_pyfunction!(py_deserialize))?;
    Ok(())
}
