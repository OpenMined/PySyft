# stdlib
from typing import Any
from typing import Type

# third party
from nacl.signing import VerifyKey

# relative
from .serde import recursive_serde_register
from .serde.serializable import serializable

recursive_serde_register(
    VerifyKey,
    serialize=lambda key: bytes(key),
    deserialize=lambda key_bytes: VerifyKey(key_bytes),
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
