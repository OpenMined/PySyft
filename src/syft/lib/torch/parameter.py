from typing import Optional, Type

from syft.proto.lib.torch.parameter_pb2 import ParameterProto
from syft.lib.torch.tensor_util import protobuf_tensor_serializer
from syft.lib.torch.tensor_util import protobuf_tensor_deserializer
from forbiddenfruit import curse
from syft.core.store.storeable_object import StorableObject

import torch as th
from torch.nn import Parameter


class PyTorchParameterWrapper(StorableObject):
    def __init__(self, value):
        super().__init__(
            data=value,
            id=value.id,
            tags=value.tags if hasattr(value, "tags") else [],
            description=value.description if hasattr(value, "description") else "",
        )
        print("Wrapped nn.Parameter with id:" + str(value.id))
        self.value = value

    def _data_object2proto(self) -> ParameterProto:
        proto = ParameterProto()
        proto.tensor.CopyFrom(protobuf_tensor_serializer(self.data))
        proto.requires_grad = self.requires_grad
        if self.grad is not None:
            proto.grad.CopyFrom(protobuf_tensor_serializer(self.grad))
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


param = type(Parameter(th.tensor([1, 2, 3])))
curse(param, "serializable_wrapper_type", PyTorchParameterWrapper)
