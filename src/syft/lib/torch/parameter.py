from typing import Optional
from typing import Type
from typing import List
from google.protobuf.reflection import GeneratedProtocolMessageType

from syft.proto.lib.torch.parameter_pb2 import ParameterProto as Parameter_PB
from syft.lib.torch.tensor_util import protobuf_tensor_serializer
from syft.lib.torch.tensor_util import protobuf_tensor_deserializer
from syft.core.store.storeable_object import StorableObject
from ...util import aggressive_set_attr
from ..generic import ObjectConstructor
from ...core.common.uid import UID

import torch as th
from torch.nn import Parameter


class ParameterConstructor(ObjectConstructor):

    # Step 1: Store the attribute name that this constructor is replacing
    constructor_name = "Parameter"

    # Step 2: Store a reference to the location on which this constructor currently lives.
    # This is also the location that this custom constructor will live once installed using
    # self.install_inside_library()
    constructor_location = th.nn.parameter

    original_type = th.nn.parameter.Parameter


# Step 3: create constructor and install it in the library
ParameterConstructor().install_inside_library()


class PyTorchParameterWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        print(
            "Wrapped torch.nn.parameter.Parameter with id:"
            + str(getattr(value, "id", ""))
        )
        self.value = value

    def _data_object2proto(self) -> Parameter_PB:
        proto = Parameter_PB()
        tensor_data = getattr(self.value, "data", None)
        if tensor_data is not None:
            proto.tensor.CopyFrom(protobuf_tensor_serializer(tensor_data))
        proto.requires_grad = getattr(self.value, "requires_grad", False)
        grad = getattr(self.value, "grad", None)
        if grad is not None:
            proto.grad.CopyFrom(protobuf_tensor_serializer(grad))
        return proto

    @staticmethod
    def _data_proto2object(proto: Parameter_PB) -> Parameter:
        data = protobuf_tensor_deserializer(proto.tensor)
        param = Parameter(data, requires_grad=proto.requires_grad)
        if proto.HasField("grad"):
            param.grad = protobuf_tensor_deserializer(proto.grad)
        return param

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return Parameter_PB

    @staticmethod
    def get_wrapped_type() -> Type:
        return Parameter

    @staticmethod
    def construct_new_object(
        id: UID,
        data: StorableObject,
        tags: Optional[List[str]],
        description: Optional[str],
    ) -> StorableObject:
        data.id = id
        data.tags = tags
        data.description = description
        return data


aggressive_set_attr(
    obj=th.nn.parameter.Parameter,
    name="serializable_wrapper_type",
    attr=PyTorchParameterWrapper,
)
