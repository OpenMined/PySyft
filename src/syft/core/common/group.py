# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from google.protobuf.empty_pb2 import Empty as Empty_PB
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft relative
from ...proto.core.auth.signed_message_pb2 import VerifyAll as VerifyAllWrapper_PB
from ...proto.core.auth.signed_message_pb2 import VerifyKey as VerifyKeyWrapper_PB
from ...util import aggressive_set_attr
from ..store.store_interface import StorableObject
from .uid import UID


class VerifyAll:
    "This class can be used to refer to the set of ALL workers."


class VerifyKeyWrapper(StorableObject):
    def __init__(self, value: VerifyKey):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> VerifyKeyWrapper_PB:
        return VerifyKeyWrapper_PB(verify_key=bytes(self.value))

    @staticmethod
    def _data_proto2object(proto: VerifyKeyWrapper_PB) -> VerifyKey:
        return VerifyKey(proto.verify_key)

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return VerifyKeyWrapper_PB

    @staticmethod
    def get_wrapped_type() -> Type:
        return VerifyKey

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
    obj=VerifyKey, name="serializable_wrapper_type", attr=VerifyKeyWrapper
)


class VerifyAllWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> VerifyAllWrapper_PB:
        return VerifyAllWrapper_PB(all=Empty_PB())

    @staticmethod
    def _data_proto2object(proto: VerifyAllWrapper_PB) -> Type:  # type: ignore
        return VerifyAll

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return VerifyAllWrapper_PB

    @staticmethod
    def get_wrapped_type() -> Type:
        return VerifyAll

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
    obj=VerifyAll, name="serializable_wrapper_type", attr=VerifyAllWrapper
)
