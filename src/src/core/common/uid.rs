use crate::core::common::serde::serializable::Serializable;
use crate::core::common::serde::serialize::serialize;
use crate::proto::core::common::Uid;
use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::PyObjectProtocol;
use std::convert::TryInto;
use std::str::FromStr;
use uuid::Uuid;

#[pyclass]
#[derive(Copy, Clone)]
#[text_signature = "(int, string, bytes, /)"]
pub struct RustUID {
    uuid: Uuid,
}

impl Serializable for RustUID {
    type ProtobufType = Uid;

    fn _object2protobuf(&self) -> Uid {
        let bytes = self.uuid.as_bytes().to_vec();

        Uid { value: bytes }
    }

    fn _protobuf2object(object: Uid) -> Self {
        let uid_format: [u8; 16] = object.value.try_into().unwrap();
        RustUID {
            uuid: Uuid::from_bytes(uid_format),
        }
    }
}

#[pymethods]
impl RustUID {
    #[new]
    pub(crate) fn new(int: Option<u128>, string: Option<&str>, bytes: Option<[u8; 16]>) -> Self {
        let uuid = if int.is_some() {
            Uuid::from_u128(int.unwrap())
        } else if string.is_some() {
            Uuid::from_str(string.unwrap()).unwrap()
        } else if bytes.is_some() {
            let bytes_repr: [u8; 16] = bytes.unwrap().into();
            Uuid::from_bytes(bytes_repr)
        } else {
            Uuid::new_v4()
        };

        RustUID { uuid }
    }

    pub(crate) fn serialize(&self, compression: bool) -> Vec<u8> {
        let bytes: Vec<u8> = serialize(self.to_owned(), compression).to_vec();
        bytes
    }

    pub fn hex(&self) -> String {
        self.uuid.to_simple().to_string()
    }

    pub fn int(&self) -> u128 {
        self.uuid.as_u128()
    }

    #[staticmethod]
    pub fn from_string(str: &str) -> Self {
        RustUID {
            uuid: Uuid::from_str(str).unwrap(),
        }
    }

    pub fn emoji(&self) -> String {
        self.__repr__()
    }

    #[getter(value)]
    fn get_value(&self) -> PyResult<u128> {
        Ok(self.int())
    }

    #[setter(value)]
    fn set_value(&mut self, value: u128) -> PyResult<()> {
        self.uuid = uuid::Uuid::from_u128(value);
        Ok(())
    }
}

#[pyproto]
impl PyObjectProtocol for RustUID {
    fn __repr__(&self) -> String {
        format!("<UID: {}>", self.hex())
    }

    fn __hash__(&self) -> u64 {
        // big problem with the 128 bits hash, I am reducing it to u64 for now but it make
        // create collision issues
        self.int() as u64
    }

    fn __richcmp__(&self, other: RustUID, op: CompareOp) -> bool {
        match op {
            CompareOp::Eq => self.uuid == other.uuid,
            CompareOp::Ne => !(self.uuid == other.uuid),
            _ => panic!("Operation not supported."),
        }
    }
}
