use crate::proto::core::common::Uid;
use crate::proto::util::DataMessage;
use bytes::BytesMut;
use prost::Message;
use pyo3::prelude::*;
use pyo3::pyclass;
use pyo3::{FromPyObject, PyAny, PyResult, Python};
use uuid::Uuid;

impl FromPyObject<'_> for Uid {
    fn extract(obj: &PyAny) -> PyResult<Self> {
        let py_proto = obj.call_method0("_object2proto").unwrap();
        let binary_repr = py_proto.call_method0("SerializeToString").unwrap();
        let bytes: &[u8] = binary_repr.extract().unwrap();
        let result: Uid = Message::decode(bytes).unwrap();
        Ok(result)
    }
}

#[pyclass]
pub(crate) struct RustUID {
    uuid: Uuid,
}

#[pymethods]
impl RustUID {
    #[new]
    fn new() -> Self {
        RustUID {
            uuid: uuid::Uuid::new_v4(),
        }
    }

    fn serialize(&mut self) -> Vec<u8> {
        let qualname = "syft.core.common.UID";
        let bytes = self.uuid.as_bytes();
        let mut mutbuf = BytesMut::new();
        let uid_msg = Uid {
            value: bytes.to_vec(),
        };
        uid_msg.encode(&mut mutbuf);

        let msg = DataMessage {
            obj_type: qualname.to_owned(),
            content: mutbuf.to_vec(),
        };

        let mut bfr = BytesMut::new();

        msg.encode(&mut bfr);

        bfr.to_vec()
    }
}

impl Uid {
    pub fn serialize(&self) -> Vec<u8> {
        let qualname = "syft.core.common.UID";
        let mut buf = BytesMut::new();
        Message::encode(self, &mut buf).unwrap();
        let msg = DataMessage {
            obj_type: qualname.to_owned(),
            content: buf.to_vec(),
        };

        let mut result = BytesMut::new();
        Message::encode(&msg, &mut result);
        result.to_vec()
    }
}
