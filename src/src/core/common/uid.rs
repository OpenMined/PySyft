use crate::core::common::serde::serializable::Serializable;
use crate::core::common::serde::serialize::serialize;
use crate::proto::core::common::Uid;
use pyo3::prelude::*;
use std::convert::TryInto;
use uuid::Uuid;

#[pyclass]
#[derive(Copy, Clone)]
pub(crate) struct RustUID {
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
    pub(crate) fn new() -> Self {
        RustUID {
            uuid: uuid::Uuid::new_v4(),
        }
    }

    fn serialize(&self, compression: bool) -> Vec<u8> {
        serialize(self.to_owned(), compression).to_vec()
    }
}
