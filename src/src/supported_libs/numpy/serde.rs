use crate::proto::lib::numpy::{DataType, TensorProto};
use crate::proto::util::DataMessage;
use numpy::npyffi::npy_float;
use prost::Message;
use pyo3::prelude::*;
use pyo3::PyResult;

//adapt this to the current Serializable Trait
#[pyfunction]
fn numpy_serde(obj: &numpy::PyArrayDyn<npy_float>) -> PyResult<Vec<u8>> {
    let readonly = obj.readonly();
    let ndim = readonly.ndim();
    let mut dims_buff: Vec<i64> = Vec::with_capacity(ndim);
    let dims = readonly.dims();
    for i in 0..ndim {
        let dim = dims[i];
        dims_buff.push(dim as i64)
    }
    let data = readonly.to_vec().unwrap();
    let numpy_internal: TensorProto = TensorProto {
        dims: dims_buff,
        byte_data: Vec::new(),
        data_type: DataType::Float as i32,
        double_data: Vec::new(),
        float_data: data,
        int32_data: Vec::new(),
        int64_data: Vec::new(),
        string_data: Vec::new(),
    };

    let mut buf: Vec<u8> = vec![];
    numpy_internal.encode(&mut buf).unwrap();

    let utils: DataMessage = DataMessage {
        obj_type: "numpy.ndarray".to_string(),
        content: buf,
    };

    let mut second_buf: Vec<u8> = vec![];
    utils.encode(&mut second_buf).unwrap();

    Ok(second_buf)
}
