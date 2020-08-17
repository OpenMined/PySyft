from typing import Optional, Type

from ..generic import ObjectConstructor
from syft.proto.lib.torch.tensor_pb2 import TensorProto
from syft.lib.torch.tensor_util import protobuf_tensor_serializer
from syft.lib.torch.tensor_util import protobuf_tensor_deserializer
from syft.core.store.storeable_object import StorableObject
from ...util import aggressive_set_attr

import torch as th


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
        self.value = value

    def _data_object2proto(self) -> TensorProto:
        proto = TensorProto()
        proto.tensor.CopyFrom(protobuf_tensor_serializer(self.value))
        if self.value.grad is not None:
            proto.grad.CopyFrom(protobuf_tensor_serializer(self.value.grad))

        return proto

    @staticmethod
    def _data_proto2object(proto: TensorProto) -> th.Tensor:
        tensor = protobuf_tensor_deserializer(proto.tensor)
        if proto.HasField("grad"):
            tensor.grad = protobuf_tensor_deserializer(proto.grad)

        return tensor

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
