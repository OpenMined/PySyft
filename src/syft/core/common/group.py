# stdlib
from typing import Any
from typing import List
from typing import Optional
from typing import Type

# third party
from google.protobuf.empty_pb2 import Empty as Empty_PB
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...proto.core.auth.signed_message_pb2 import VerifyAll as VerifyAllWrapper_PB
from ...proto.core.auth.signed_message_pb2 import VerifyKey as VerifyKey_PB
from ..store.store_interface import StorableObject
from .serde.serializable import Serializable
from .serde.serializable import bind_protobuf
from .uid import UID


def object2proto(obj: Any) -> VerifyKey_PB:
    return VerifyKey_PB(verify_key=bytes(obj))


def proto2object(proto: VerifyKey_PB) -> VerifyKey:
    return VerifyKey(proto.verify_key)


GenerateWrapper(
    wrapped_type=VerifyKey,
    import_path="nacl.signing.VerifyKey",
    protobuf_scheme=VerifyKey_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)


@bind_protobuf
class VerifyAll(Serializable):
    def _object2proto(self) -> VerifyAllWrapper_PB:
        return VerifyAllWrapper_PB(all=Empty_PB())

    @staticmethod
    def _proto2object(proto: VerifyAllWrapper_PB) -> Type:  # type: ignore
        return VerifyAll

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return VerifyAllWrapper_PB

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
