from typing import Optional, Type

from ..generic import ObjectConstructor
from syft.proto.lib.torch.tensor_pb2 import TensorProto
from syft.core.store.storeable_object import StorableObject
from ...util import aggressive_set_attr

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


class UppercaseTensorConstructor(ObjectConstructor):

    __name__ = "UppercaseTensorConstructor"

    # Step 1: Store the attribute name that this constructor is replacing
    constructor_name = "Tensor"

    # Step 2: Store a reference to the location on which this constructor currently lives.
    # This is also the location that this custom constructor will live once installed using
    # self.install_inside_library()
    constructor_location = th

    original_type = th.Tensor


# Step 3: create constructor and install it in the library
UppercaseTensorConstructor().install_inside_library()

torch_tensor_type = type(th.tensor([1, 2, 3]))


class TorchTensorWrapper(StorableObject):
    def __init__(self, value):
        super().__init__(
            data=value,
            id=value.id,
            tags=value.tags if hasattr(value, "tags") else [],
            description=value.description if hasattr(value, "description") else "",
        )
        print("Wrapped tensor with id:" + str(value.id))
        self.value = value

    def _data_object2proto(self) -> TensorProto:
        tensor = self.value
        proto = TensorProto()

        dtype = TORCH_DTYPE_STR[tensor.dtype]
        if tensor.is_quantized:
            proto.is_quantized = True
            proto.scale = tensor.q_scale()
            proto.zero_point = tensor.q_zero_point()
            data = tensor.flatten().int_repr().tolist()
        else:
            data = tensor.flatten().tolist()

        proto.dtype = dtype
        proto.shape.extend(tensor.size())
        getattr(proto, "contents_" + dtype).extend(data)

        return proto

    @staticmethod
    def _data_proto2object(proto: TensorProto) -> int:
        size = tuple(proto.shape)
        data = getattr(proto, "contents_" + proto.dtype)

        if proto.is_quantized:
            # Drop the 'q' from the beginning of the quantized dtype to get the int type
            dtype = TORCH_STR_DTYPE[proto.dtype[1:]]
            int_tensor = th.tensor(data, dtype=dtype).reshape(size)
            # Automatically converts int types to quantized types
            return th._make_per_tensor_quantized_tensor(
                int_tensor, proto.scale, proto.zero_point
            )
        else:
            dtype = TORCH_STR_DTYPE[proto.dtype]
            return th.tensor(data, dtype=dtype).reshape(size)

    @staticmethod
    def get_data_protobuf_schema() -> Optional[Type]:
        return TensorProto

    @staticmethod
    def get_wrapped_type() -> type:
        return torch_tensor_type

    @staticmethod
    def construct_new_object(id, data, tags, description):
        data.id = id
        data.tags = tags
        data.description = description
        return data


aggressive_set_attr(
    obj=torch_tensor_type, name="serializable_wrapper_type", attr=TorchTensorWrapper
)
