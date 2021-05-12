use crate::core::common::serde::serializable::Serializable;
use crate::core::common::uid::RustUID;
use bytes::BytesMut;
use pyo3::prelude::*;
use pyo3::PyClass;

pub(crate) fn deserialize<T>(bytes: BytesMut) -> T
where
    T: Serializable,
{
    Serializable::_deserialize(bytes)
}
