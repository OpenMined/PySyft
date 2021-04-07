# stdlib
from typing import Any
from typing import Optional
from typing import Union

# syft relative
from ...core.common import UID
from ...proto.lib.python.range_pb2 import Range as Range_PB
from .primitive_factory import PyPrimitive


class Range(PyPrimitive):
    def __init__(
        self,
        start: Any = None,
        stop: Union[Any] = None,
        step: Union[Any] = None,
        id: Optional[UID] = None,
    ):
        if stop is None and step is None:
            stop = start
            start = None

        self.value = range(start, stop, step)
        self._id = id if id else UID()

    @property
    def id(self) -> UID:
        return self._id

    def _object2proto(self) -> Range_PB:
        pass

    @staticmethod
    def _proto2object(proto: Range_PB) -> "Range":
        pass
