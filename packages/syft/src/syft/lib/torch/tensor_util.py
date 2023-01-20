# third party
import pyarrow as pa
import torch as th

# relative
from ...experimental_flags import flags
from ...proto.lib.torch.tensor_pb2 import ProtobufContent
from ...proto.lib.torch.tensor_pb2 import TensorData

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


def protobuf_data_encoding(tensor: th.Tensor) -> bytes:
    protobuf_tensor_data = ProtobufContent()

    if tensor.is_quantized:
        data = th.flatten(tensor).int_repr().tolist()
    else:
        data = th.flatten(tensor).tolist()

    dtype = TORCH_DTYPE_STR[tensor.dtype]
    protobuf_tensor_data.shape.extend(tensor.size())
    getattr(protobuf_tensor_data, "contents_" + dtype).extend(data)
    return protobuf_tensor_data.SerializeToString()


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


def tensor_serializer(tensor: th.Tensor) -> TensorData:
    """Strategy to serialize a tensor using Protobuf"""

    protobuf_tensor = TensorData()

    if tensor.is_quantized:
        protobuf_tensor.is_quantized = True
        protobuf_tensor.scale = tensor.q_scale()
        protobuf_tensor.zero_point = tensor.q_zero_point()

    if flags.APACHE_ARROW_TENSOR_SERDE:
        protobuf_tensor.arrow_data = arrow_data_encoding(tensor)
    else:
        protobuf_tensor.proto_data = protobuf_data_encoding(tensor)

    protobuf_tensor.dtype = TORCH_DTYPE_STR[tensor.dtype]
    return protobuf_tensor.SerializeToString()


def protobuf_data_decoding(protobuf_tensor: TensorData) -> th.Tensor:
    proto_data = ProtobufContent()
    proto_data.ParseFromString(protobuf_tensor.proto_data)
    size = tuple(proto_data.shape)
    data = getattr(proto_data, "contents_" + protobuf_tensor.dtype)

    if protobuf_tensor.is_quantized:
        # Drop the 'q' from the beginning of the quantized dtype to get the int type
        dtype = TORCH_STR_DTYPE[protobuf_tensor.dtype[1:]]
        int_tensor = th.tensor(data, dtype=dtype).reshape(size)
        # Automatically converts int types to quantized types
        return th._make_per_tensor_quantized_tensor(
            int_tensor, protobuf_tensor.scale, protobuf_tensor.zero_point
        )
    else:
        dtype = TORCH_STR_DTYPE[protobuf_tensor.dtype]
        return th.tensor(data, dtype=dtype).reshape(size)


def arrow_data_decoding(tensor_data: TensorData) -> th.Tensor:
    reader = pa.BufferReader(tensor_data.arrow_data)
    buf = reader.read_buffer()
    result = pa.ipc.read_tensor(buf)
    np_array = result.to_numpy()
    np_array.setflags(write=True)
    data = th.from_numpy(np_array)

    if tensor_data.is_quantized:
        result = th._make_per_tensor_quantized_tensor(
            data, tensor_data.scale, tensor_data.zero_point
        )
    else:
        result = data

    if tensor_data.dtype == "bfloat16":
        result = result.type(th.bfloat16).clone()

    if tensor_data.dtype == "bool":
        result = result.type(th.bool).clone()

    return result


def tensor_deserializer(buf: bytes) -> th.Tensor:
    protobuf_tensor = TensorData()
    protobuf_tensor.ParseFromString(buf)

    if protobuf_tensor.HasField("arrow_data"):
        return arrow_data_decoding(protobuf_tensor)
    elif protobuf_tensor.HasField("proto_data"):
        return protobuf_data_decoding(protobuf_tensor)
