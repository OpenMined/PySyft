# stdlib
from typing import Any
from typing import Type

# third party
from nacl.signing import VerifyKey

# relative
from ...proto.core.auth.signed_message_pb2 import VerifyKey as VerifyKey_PB
from .serde.serializable import serializable


def object2proto(obj: Any) -> VerifyKey_PB:
    return VerifyKey_PB(verify_key=bytes(obj))


def proto2object(proto: VerifyKey_PB) -> VerifyKey:
    return VerifyKey(proto.verify_key)


serializable(generate_wrapper=True)(
    wrapped_type=VerifyKey,
    import_path="nacl.signing.VerifyKey",
    protobuf_scheme=VerifyKey_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)


def _create_VERIFYALL() -> Any:
    @serializable(recursive_serde=True)
    class VerifyAll:
        __attr_allowlist__ = []
        _instance = None

        def __new__(cls: Type) -> "VerifyAll":
            if cls._instance is None:
                cls._instance = object.__new__(cls)
            return cls._instance

    return VerifyAll()


VERIFYALL = _create_VERIFYALL()
VerifyAll = type(
    VERIFYALL
)  # deprecated: https://github.com/OpenMined/PySyft/issues/5396
