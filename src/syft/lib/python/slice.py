# stdlib
import sys
from typing import Any
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ... import deserialize
from ... import serialize
from ...core.common import UID
from ...core.common.serde.serializable import bind_protobuf
from ...logger import traceback_and_raise
from ...proto.lib.python.slice_pb2 import Slice as Slice_PB
from .primitive_factory import PrimitiveFactory
from .primitive_factory import isprimitive
from .primitive_interface import PyPrimitive
from .types import SyPrimitiveRet


@bind_protobuf
class Slice(PyPrimitive):
    def __init__(
        self,
        start: Any = None,
        stop: Any = None,
        step: Optional[Any] = None,
        id: Optional[UID] = None,
    ):
        self._slice = slice(start, stop, step)
        self.start = self._slice.start
        self.stop = self._slice.stop
        self.step = self._slice.step
        self._id: UID = id if id else UID()

    @property
    def id(self) -> UID:
        """We reveal PyPrimitive.id as a property to discourage users and
        developers of Syft from modifying .id attributes after an object
        has been initialized.

        :return: returns the unique id of the object
        :rtype: UID
        """
        return self._id

    def upcast(self) -> slice:
        return slice(self)

    def __eq__(self, other: Any) -> SyPrimitiveRet:
        res = super().__eq__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __ne__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ne__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __sizeof__(self) -> SyPrimitiveRet:
        res = super().__sizeof__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __getitem__(self, key: Any) -> Any:
        res = key
        # we might be holding a primitive value, but generate_primitive
        # doesn't handle non primitives so we should check
        if isprimitive(value=res):
            return PrimitiveFactory.generate_primitive(value=res)
        return res

    def getindices(
        self,
        length: Any = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> int:
        if self.step is None:
            self.step = 1
        if not isinstance(step, int):
            traceback_and_raise(TypeError("expected integer type arguments for step"))
        else:
            self.step = step
        if self.start is None:
            self.start = length - 1 if step < 0 else 0
        if not isinstance(start, int):
            traceback_and_raise(TypeError("expected integer type arguments for start"))
        else:
            self.start = start
            if start < 0:
                self.start = start + length
        if self.stop is None:
            self.stop = -1 if step < 0 else length
        if not isinstance(stop, int):
            traceback_and_raise(TypeError("expected integer type arguments for stop"))
        else:
            self.stop = stop
            if stop < 0:
                self.stop = stop + length
        if stop > length:
            traceback_and_raise(
                TypeError("no attributes should be more than maximum length")
            )
        if start >= length:
            traceback_and_raise(
                TypeError("no attributes should be more than maximum length")
            )
        if step == 0:
            traceback_and_raise(
                TypeError("no attributes should be more than maximum length")
            )
        return 0

    def unpack(self, start: Any, stop: Any, step: Any) -> int:
        if self.step is None:
            self.step = 1
        if step == 0:
            traceback_and_raise(TypeError("slice step can't be zero"))
        if step < -sys.maxsize:
            self.step = -sys.maxsize
        if self.start is None:
            self.start = sys.maxsize if step < 0 else 0

        if self.stop is None:
            self.stop = -sys.maxsize - 1 if step < 0 else sys.maxsize
        return 0

    def adjustindices(self, length: Any, start: Any, stop: Any, step: Any) -> int:
        if start < 0:
            self.start = start + length
            if start < 0:
                self.start = -1 if step < 0 else 0
        else:
            if start >= length:
                self.start = length - 1 if step < 0 else length

        if stop < 0:
            self.stop = stop + length
            if stop < 0:
                self.stop = -1 if step < 0 else 0

        else:
            if stop >= length:
                self.stop = length - 1 if step < 0 else length

        if step < 0:
            if stop < start:
                return (start - stop - 1) / (-step) + 1

        else:
            if start < stop:
                return (stop - start - 1) / step + 1
        return 0

    def getindicesex(self, length: Any, start: Any, stop: Any, step: Any) -> int:
        if self.unpack(start, stop, step) < 0:
            traceback_and_raise(TypeError("invalid values"))
        return 0

    def _object2proto(self) -> Slice_PB:
        return Slice_PB(
            start=self.start,
            stop=self.stop,
            step=self.step,
            id=serialize(obj=self._id),
        )

    @staticmethod
    def _proto2object(proto: Slice_PB) -> "Slice":
        id_: UID = deserialize(blob=proto.id)

        return Slice(
            start=proto.start,
            stop=proto.stop,
            step=proto.step,
            id=id_,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Slice_PB
