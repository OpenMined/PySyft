# stdlib
from typing import Any

# relative
from ...core.common import UID
from ...core.common.serde.serializable import Serializable


class PyPrimitive(Serializable):
    def __init__(self, temp_storage_for_actual_primitive: bool = False) -> None:
        self._id: UID

        # sometimes we need to serialize a python primitive in such a way that we can
        # deserialize it back as that primitive later. This flag allows us to do that.
        self.temp_storage_for_actual_primitive = temp_storage_for_actual_primitive

    def upcast(self) -> Any:
        pass
