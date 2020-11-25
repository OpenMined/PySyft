# stdlib
import dataclasses
from typing import List as TypedList
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from sympc.config import Config
from sympc.tensor import FixedPrecisionTensor

# syft relative
from ...core.common import UID
from ...core.store.storeable_object import StorableObject
from ...lib.torch.tensor_util import protobuf_tensor_deserializer
from ...lib.torch.tensor_util import protobuf_tensor_serializer
from ...proto.lib.sympc.fixed_precision_tensor_pb2 import (
    FixedPrecisionTensor as FixedPrecisionTensor_PB,
)
from ...util import aggressive_set_attr
from ..python import Dict
from ..python.primitive_factory import PrimitiveFactory


class SyFixedPrecisionTensorWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> FixedPrecisionTensor_PB:

        fpt: FixedPrecisionTensor = self.value

        conf = PrimitiveFactory.generate_primitive(value=dataclasses.asdict(fpt.config))

        # Remove redundant data
        conf.pop("max_value")
        conf.pop("min_value")

        conf = conf._object2proto()

        proto = FixedPrecisionTensor_PB(config=conf)

        tensor_data = getattr(fpt._tensor, "data", None)
        if tensor_data is not None:
            proto.tensor.tensor.CopyFrom(protobuf_tensor_serializer(tensor_data))
        proto.tensor.requires_grad = getattr(fpt._tensor, "requires_grad", False)
        grad = getattr(fpt._tensor, "grad", None)
        if grad is not None:
            proto.tensor.grad.CopyFrom(protobuf_tensor_serializer(grad))

        return proto

    @staticmethod
    def _data_proto2object(
        proto: FixedPrecisionTensor_PB,
    ) -> FixedPrecisionTensor:

        conf_dict = Dict._proto2object(proto=proto.config)
        conf_dict = {key.data: value for key, value in conf_dict.items()}
        conf = Config(**conf_dict)

        data = protobuf_tensor_deserializer(proto.tensor.tensor)
        fpt = FixedPrecisionTensor(data=None, config=conf)

        # Manually put the tensor since we do not want to re-encode it
        fpt._tensor = data

        return fpt

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return FixedPrecisionTensor_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return FixedPrecisionTensor

    @staticmethod
    def construct_new_object(
        id: UID,
        data: StorableObject,
        description: Optional[str],
        tags: Optional[TypedList[str]],
    ) -> StorableObject:
        data.id = id
        data.tags = tags
        data.description = description
        return data


aggressive_set_attr(
    obj=FixedPrecisionTensor,
    name="serializable_wrapper_type",
    attr=SyFixedPrecisionTensorWrapper,
)
