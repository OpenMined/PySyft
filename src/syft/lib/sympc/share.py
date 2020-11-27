# stdlib
from typing import List as TypedList
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from sympc.tensor import ShareTensor

# syft relative
from ...core.common import UID
from ...core.store.storeable_object import StorableObject
from ...lib.torch.tensor_util import protobuf_tensor_deserializer
from ...lib.torch.tensor_util import protobuf_tensor_serializer
from ...proto.lib.sympc.share_tensor_pb2 import ShareTensor as ShareTensor_PB
from ...util import aggressive_set_attr
from .session_util import protobuf_session_deserializer
from .session_util import protobuf_session_serializer


class SyShareTensorWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> ShareTensor_PB:

        share: ShareTensor = self.value

        session = protobuf_session_serializer(share.session)
        proto = ShareTensor_PB(session=session)

        tensor_data = getattr(share.tensor, "data", None)
        if tensor_data is not None:
            proto.tensor.tensor.CopyFrom(protobuf_tensor_serializer(tensor_data))
        proto.tensor.requires_grad = getattr(share.tensor, "requires_grad", False)
        grad = getattr(share.tensor, "grad", None)
        if grad is not None:
            proto.tensor.grad.CopyFrom(protobuf_tensor_serializer(grad))

        return proto

    @staticmethod
    def _data_proto2object(proto: ShareTensor_PB) -> ShareTensor:
        session = protobuf_session_deserializer(proto=proto.session)

        data = protobuf_tensor_deserializer(proto.tensor.tensor)
        share = ShareTensor(data=None, session=session)

        # Manually put the tensor since we do not want to re-encode it
        share.tensor = data.type(session.tensor_type)

        return share

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return ShareTensor_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return ShareTensor

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
    obj=ShareTensor,
    name="serializable_wrapper_type",
    attr=SyShareTensorWrapper,
)
