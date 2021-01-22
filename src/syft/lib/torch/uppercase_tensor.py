# stdlib
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import torch as th
import warnings

# syft relative
from ...core.common.uid import UID
from ...core.store.storeable_object import StorableObject
from ...lib.torch.tensor_util import protobuf_tensor_deserializer
from ...lib.torch.tensor_util import protobuf_tensor_serializer
from ...proto.lib.torch.tensor_pb2 import TensorProto as Tensor_PB
from ...util import aggressive_set_attr
from ...proto.lib.torch.device_pb2 import Device as Device_PB

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

        proto.requires_grad = getattr(self.value, "requires_grad", False)
        proto.device.CopyFrom(
            Device_PB(
                type=self.value.device.type,  # type: ignore
                index=self.value.device.index,  # type: ignore
            )
        )

        if proto.requires_grad:
            grad = getattr(self.value, "grad", None)
            if grad is not None:
                proto.grad.CopyFrom(protobuf_tensor_serializer(grad))

        return proto

    @staticmethod
    def _data_proto2object(proto: Tensor_PB) -> th.Tensor:
        tensor = protobuf_tensor_deserializer(proto.tensor)
        if proto.HasField("grad"):
            tensor.grad = protobuf_tensor_deserializer(proto.grad)

        tensor.requires_grad_(proto.requires_grad)

        if proto.device.type == "cuda" and th.cuda.is_available():
            cuda_index = proto.device.index
            if th.cuda.device_count() < (cuda_index + 1):
                cuda_index = th.cuda.device_count() - 1
                warnings.warn(
                    f"The cuda index in message is {cuda_index}, it's outof range. "
                    + "Using the last of your available GPU indexex."
                )
            return tensor.cuda(cuda_index)
        else:
            warnings.warn(
                "The device in message is 'cuda', but there is no GPU available. Using CPU."
            )
            return tensor

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return Tensor_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return torch_tensor_type

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
    obj=torch_tensor_type, name="serializable_wrapper_type", attr=TorchTensorWrapper
)
