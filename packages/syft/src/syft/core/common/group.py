# stdlib
from typing import Any
from typing import Type

# third party
from google.protobuf.empty_pb2 import Empty as Empty_PB
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...proto.core.auth.signed_message_pb2 import VerifyAll as VerifyAllWrapper_PB
from ...proto.core.auth.signed_message_pb2 import VerifyKey as VerifyKey_PB
from .serde.serializable import Serializable
from .serde.serializable import bind_protobuf


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


def _create_VERIFYALL() -> Any:
    @bind_protobuf
    class VerifyAll(Serializable):
        _instance = None

        def __new__(cls: Type) -> "VerifyAll":
            if cls._instance is None:
                cls._instance = object.__new__(cls)
            return cls._instance

        def _object2proto(self) -> VerifyAllWrapper_PB:
            return VerifyAllWrapper_PB(all=Empty_PB())

        @staticmethod
        def _proto2object(proto: VerifyAllWrapper_PB) -> "VerifyAll":
            return VERIFYALL

        @staticmethod
        def get_protobuf_schema() -> GeneratedProtocolMessageType:
            return VerifyAllWrapper_PB

    return VerifyAll()


VERIFYALL = _create_VERIFYALL()
VerifyAll = type(
    VERIFYALL
)  # deprecated: https://github.com/OpenMined/PySyft/issues/5396
