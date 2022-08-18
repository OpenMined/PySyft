# third party
import pyarrow as pa
import torch as th

from ...core.common.serde import _serialize, _deserialize

# Torch dtypes to string (and back) mappers
TORCH_DTYPE_STR = {
    th.uint8: "uint8",
    th.int8: "int8",
    th.int16: "int16",
    th.int32: "int32",
    th.int64: "int64",
    th.float16: "float16",
    th.float32: "float32",
    th.float64: "float64",
    # th.complex32: "complex32", # deprecated
    # th.complex64: "complex64",
    # th.complex128: "complex128",
    th.bool: "bool",
    th.qint8: "qint8",
    th.quint8: "quint8",
    th.qint32: "qint32",
    th.bfloat16: "bfloat16",
}
TORCH_STR_DTYPE = {name: cls for cls, name in TORCH_DTYPE_STR.items()}


def arrow_data_encoding(tensor: th.Tensor) -> bytes:
    if TORCH_DTYPE_STR[tensor.dtype] == "bfloat16":
        tensor = tensor.type(th.float32)

    if tensor.is_quantized:
        numpy_tensor = tensor.detach().int_repr().numpy()
    else:
        numpy_tensor = tensor.detach().numpy()

    apache_arrow = pa.Tensor.from_numpy(obj=numpy_tensor)
    sink = pa.BufferOutputStream()
    pa.ipc.write_tensor(apache_arrow, sink)
    return sink.getvalue().to_pybytes()


def tensor_serializer(tensor: th.Tensor) -> bytes:
    """Strategy to serialize a tensor using Protobuf"""

    arrow_data = arrow_data_encoding(tensor)

    if tensor.is_quantized:
        is_quantized = True
        scale = tensor.q_scale()
        zero_point = tensor.q_zero_point()
    else:
        is_quantized = False
        scale = None
        zero_point = None

    dtype = TORCH_DTYPE_STR[tensor.dtype]
    return _serialize((arrow_data, dtype, is_quantized, scale, zero_point), to_bytes=True)


def arrow_data_decoding(data, dtype, is_quantized, scale, zero_point) -> th.Tensor:
    reader = pa.BufferReader(data)
    buf = reader.read_buffer()
    result = pa.ipc.read_tensor(buf)
    np_array = result.to_numpy()
    np_array.setflags(write=True)
    data = th.from_numpy(np_array)

    if is_quantized:
        result = th._make_per_tensor_quantized_tensor(
            data, scale, zero_point
        )
    else:
        result = data

    if dtype == "bfloat16":
        result = result.type(th.bfloat16).clone()

    if dtype == "bool":
        result = result.type(th.bool).clone()

    return result


def tensor_deserializer(buf: bytes) -> th.Tensor:
    (data, dtype, is_quantized, scale, zero_point) = _deserialize(buf, from_bytes=True)
    return arrow_data_decoding(data, dtype, is_quantized, scale, zero_point)
