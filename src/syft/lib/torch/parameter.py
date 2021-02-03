# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import torch as th
from torch.nn import Parameter

# syft relative
from ...core.common.uid import UID
from ...core.store.storeable_object import StorableObject
from ...lib.torch.tensor_util import protobuf_tensor_deserializer
from ...lib.torch.tensor_util import protobuf_tensor_serializer
from ...proto.lib.torch.parameter_pb2 import ParameterProto as Parameter_PB
from ...util import aggressive_set_attr

torch_tensor = th.tensor([1.0, 2.0, 3.0])
torch_parameter_type = type(th.nn.parameter.Parameter(torch_tensor))


class PyTorchParameterWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
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

        # opacus monkey patches this onto the Parameter class
        grad_sample = getattr(self.value, "grad_sample", None)
        if grad_sample is not None:
            proto.grad_sample.CopyFrom(protobuf_tensor_serializer(grad_sample))
        return proto

    @staticmethod
    def _data_proto2object(proto: Parameter_PB) -> Parameter:
        data = protobuf_tensor_deserializer(proto.tensor)
        param = Parameter(data, requires_grad=proto.requires_grad)
        if proto.HasField("grad"):
            param.grad = protobuf_tensor_deserializer(proto.grad)

        # opacus monkey patches this onto the Parameter class
        if proto.HasField("grad_sample"):
            param.grad_sample = protobuf_tensor_deserializer(proto.grad_sample)
        return param

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return Parameter_PB

    @staticmethod
    def get_wrapped_type() -> Type:
        return torch_parameter_type

    @staticmethod
    def construct_new_object(
        id: UID,
        data: StorableObject,
        description: Optional[str],
        tags: Optional[List[str]],
    ) -> StorableObject:
        data.id = id
        data.tags = tags
        data.description = description
        return data


aggressive_set_attr(
    obj=torch_parameter_type,
    name="serializable_wrapper_type",
    attr=PyTorchParameterWrapper,
)
