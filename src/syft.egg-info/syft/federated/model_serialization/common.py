# stdlib
from typing import Any
from typing import Union

# third party
from syft_proto.types.torch.v1.tensor_data_pb2 import TensorData as TensorData_PB
from syft_proto.types.torch.v1.tensor_pb2 import TorchTensor as TorchTensor_PB
import torch as th

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


def set_protobuf_id(field: Any, id: Union[int, str]) -> None:
    if isinstance(id, str):
        field.id_str = id
    else:
        field.id_int = id


def get_protobuf_id(field: Any) -> Union[int, str]:
    return getattr(field, field.WhichOneof("id"))


def serialize_tensor(tensor: th.Tensor) -> TorchTensor_PB:
    """
    This method converts a Torch tensor into a serialized tensor
    using Protobuf.

    Args:
        tensor (th.Tensor): an input tensor to be serialized

    Returns:
        protobuf_obj: Protobuf version of torch tensor.
    """
    dtype = TORCH_DTYPE_STR[tensor.dtype]

    tensor_data = TensorData_PB()

    if tensor.is_quantized:
        tensor_data.is_quantized = True
        tensor_data.scale = tensor.q_scale()
        tensor_data.zero_point = tensor.q_zero_point()
        data = th.flatten(tensor).int_repr().tolist()
    else:
        data = th.flatten(tensor).tolist()

    tensor_data.dtype = dtype
    tensor_data.shape.dims.extend(tensor.size())
    getattr(tensor_data, "contents_" + dtype).extend(data)

    protobuf_tensor = TorchTensor_PB()
    set_protobuf_id(protobuf_tensor.id, getattr(tensor, "id", 1))

    protobuf_tensor.serializer = TorchTensor_PB.Serializer.SERIALIZER_ALL
    protobuf_tensor.contents_data.CopyFrom(tensor_data)
    protobuf_tensor.tags.extend(getattr(tensor, "tags", []))
    return protobuf_tensor


def deserialize_tensor(protobuf_tensor: TorchTensor_PB) -> th.Tensor:
    """
    This method converts a Protobuf torch tensor back into a
    Torch tensor.

    Args:
        protobuf_tensor (bin): Protobuf message of torch tensor.

    Returns:
        tensor (th.Tensor): a torch tensor converted from Protobuf
    """
    tensor_id = get_protobuf_id(protobuf_tensor.id)
    tags = protobuf_tensor.tags
    description = protobuf_tensor.description

    contents_type = protobuf_tensor.WhichOneof("contents")
    tensor_data_pb = getattr(protobuf_tensor, contents_type)

    size = tuple(tensor_data_pb.shape.dims)
    data = getattr(tensor_data_pb, "contents_" + tensor_data_pb.dtype)

    if tensor_data_pb.is_quantized:
        # Drop the 'q' from the beginning of the quantized dtype to get the int type
        dtype = TORCH_STR_DTYPE[tensor_data_pb.dtype[1:]]
        int_tensor = th.tensor(data, dtype=dtype).reshape(size)
        # Automatically converts int types to quantized types
        tensor = th._make_per_tensor_quantized_tensor(
            int_tensor, tensor_data_pb.scale, tensor_data_pb.zero_point
        )
    else:
        dtype = TORCH_STR_DTYPE[tensor_data_pb.dtype]
        tensor = th.tensor(data, dtype=dtype).reshape(size)

    tensor.id = tensor_id
    tensor.tags = set(tags)
    tensor.description = description

    return tensor
