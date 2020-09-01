# third party
import torch as th

# syft relative
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
    th.complex32: "complex32",
    th.complex64: "complex64",
    th.complex128: "complex128",
    th.bool: "bool",
    th.qint8: "qint8",
    th.quint8: "quint8",
    th.qint32: "qint32",
    th.bfloat16: "bfloat16",
}
TORCH_STR_DTYPE = {name: cls for cls, name in TORCH_DTYPE_STR.items()}


def protobuf_tensor_serializer(tensor: th.Tensor) -> TensorData:
    """Strategy to serialize a tensor using Protobuf"""
    dtype = TORCH_DTYPE_STR[tensor.dtype]

    protobuf_tensor = TensorData()

    if tensor.is_quantized:
        protobuf_tensor.is_quantized = True
        protobuf_tensor.scale = tensor.q_scale()
        protobuf_tensor.zero_point = tensor.q_zero_point()
        data = th.flatten(tensor).int_repr().tolist()
    else:
        data = th.flatten(tensor).tolist()

    protobuf_tensor.dtype = dtype
    protobuf_tensor.shape.extend(tensor.size())
    getattr(protobuf_tensor, "contents_" + dtype).extend(data)

    return protobuf_tensor


def protobuf_tensor_deserializer(protobuf_tensor: TensorData) -> th.Tensor:
    """Strategy to deserialize a binary input using Protobuf"""
    size = tuple(protobuf_tensor.shape)
    data = getattr(protobuf_tensor, "contents_" + protobuf_tensor.dtype)

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
