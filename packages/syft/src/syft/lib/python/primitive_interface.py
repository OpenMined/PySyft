# stdlib
from typing import Any

# relative
from ...core.common import UID


class PyPrimitive:
    def __init__(self, temporary_box: bool = False) -> None:
        self._id: UID

        # sometimes we need to serialize a python primitive in such a way that we can
        # deserialize it back as that primitive later. This flag allows us to do that.
        self.temporary_box = temporary_box

    def upcast(self) -> Any:
        pass
