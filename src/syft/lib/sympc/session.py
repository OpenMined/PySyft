# stdlib
from typing import List as TypedList
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from sympc.session import Session

# syft relative
from ...core.common import UID
from ...core.store.storeable_object import StorableObject
from ...proto.lib.sympc.session_pb2 import MPCSession as MPCSession_PB
from ...util import aggressive_set_attr
from .session_util import protobuf_session_deserializer
from .session_util import protobuf_session_serializer


class SySessionWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> MPCSession_PB:
        proto = protobuf_session_serializer(self.value)
        return proto

    @staticmethod
    def _data_proto2object(proto: MPCSession_PB) -> "SySessionWrapper":
        session = protobuf_session_deserializer(proto)
        return session

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return MPCSession_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return Session

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
    obj=Session, name="serializable_wrapper_type", attr=SySessionWrapper
)
