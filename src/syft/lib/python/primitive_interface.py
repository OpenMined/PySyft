# stdlib
from typing import Any

# syft relative
from ...core.common import UID
from ...core.common.serde.serializable import Serializable


class PyPrimitive(Serializable):
    def __init__(self) -> None:
        self._id: UID

    def upcast(self) -> Any:
        pass
