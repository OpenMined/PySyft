from typing import Optional
from typing import List
from google.protobuf.reflection import GeneratedProtocolMessageType

from ..generic import ObjectConstructor
from syft.proto.lib.torch.tensor_pb2 import TensorProto as Tensor_PB
from syft.lib.torch.tensor_util import protobuf_tensor_serializer
from syft.lib.torch.tensor_util import protobuf_tensor_deserializer
from syft.core.store.storeable_object import StorableObject
from ...util import aggressive_set_attr
from ...core.common.uid import UID

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
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> Tensor_PB:
        proto = Tensor_PB()
        proto.tensor.CopyFrom(protobuf_tensor_serializer(self.value))

        grad = getattr(self.value, "grad", None)
        if grad is not None:
            proto.grad.CopyFrom(protobuf_tensor_serializer(grad))

        return proto

    @staticmethod
    def _data_proto2object(proto: Tensor_PB) -> th.Tensor:
        tensor = protobuf_tensor_deserializer(proto.tensor)
        if proto.HasField("grad"):
            tensor.grad = protobuf_tensor_deserializer(proto.grad)

        return tensor

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return Tensor_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return torch_tensor_type

    @staticmethod
    def construct_new_object(
        id: UID, data: StorableObject, tags: List[str], description: Optional[str]
    ) -> object:
        data.id = id
        data.tags = tags
        data.description = description
        return data


aggressive_set_attr(
    obj=torch_tensor_type, name="serializable_wrapper_type", attr=TorchTensorWrapper
)
