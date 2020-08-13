from typing import Optional, Type

from syft.proto.lib.torch.parameter_pb2 import ParameterProto
from syft.lib.torch.tensor_util import protobuf_tensor_serializer
from syft.lib.torch.tensor_util import protobuf_tensor_deserializer
from syft.core.store.storeable_object import StorableObject
from ...util import aggressive_set_attr
from ..generic import ObjectConstructor

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
    def __init__(self, value):
        super().__init__(
            data=value,
            id=value.id,
            tags=value.tags if hasattr(value, "tags") else [],
            description=value.description if hasattr(value, "description") else "",
        )
        print("Wrapped torch.nn.parameter.Parameter with id:" + str(value.id))
        self.value = value

    def _data_object2proto(self) -> ParameterProto:
        proto = ParameterProto()
        proto.tensor.CopyFrom(protobuf_tensor_serializer(self.value.data))
        proto.requires_grad = self.value.requires_grad
        if self.value.grad is not None:
            proto.grad.CopyFrom(protobuf_tensor_serializer(self.value.grad))
        return proto

    @staticmethod
    def _data_proto2object(proto: ParameterProto) -> Parameter:
        data = protobuf_tensor_deserializer(proto.tensor)
        param = Parameter(data, requires_grad=proto.requires_grad)
        if proto.HasField("grad"):
            param.grad = protobuf_tensor_deserializer(proto.grad)
        return param

    @staticmethod
    def get_data_protobuf_schema() -> Optional[Type]:
        return ParameterProto

    @staticmethod
    def get_wrapped_type() -> type:
        return Parameter

    @staticmethod
    def construct_new_object(id, data, tags, description):
        data.id = id
        data.tags = tags
        data.description = description
        return data


param_type = type(Parameter(th.tensor([1.0, 2, 3])))
aggressive_set_attr(
    obj=param_type, name="serializable_wrapper_type", attr=PyTorchParameterWrapper
)
