# stdlib
import sys
from typing import Any
from typing import Optional
from typing import Union

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
class Slice(slice, PyPrimitive):
    def __init__(
        self,
        start: Any = None,
        stop: Any = None,
        step: Any = None,
        id: Optional[UID] = None,
    ):
        if not start and not stop and not step:
            self.start = None
            self.stop = None
            self.step = None
        super().__init__()
        self._id: UID = id if id else UID()

    def __eq__(self, other: Any) -> SyPrimitiveRet:
        res = super().__eq__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __ge__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ge__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __gt__(self, other: Any) -> SyPrimitiveRet:
        res = super().__gt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __hash__(self) -> SyPrimitiveRet:
        res = super().__hash__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __le__(self, other: Any) -> SyPrimitiveRet:
        res = super().__le__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __lt__(self, other: Any) -> SyPrimitiveRet:
        res = super().__lt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __ne__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ne__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __sizeof__(self) -> SyPrimitiveRet:
        res = super().__sizeof__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __getitem__(self, key: Union[int, slice]) -> Any:
        res = super().__getitem__(key)  # ignore = type
        if isprimitive(value=res):
            return PrimitiveFactory.generate_primitive(value=res)
        return res

    def copy(self) -> "Slice":
        res = super().copy()
        res._id = UID()
        return res

    def getindices(
        self,
        length: Any = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> SyPrimitiveRet:
        if self.step is None:
            step = 1
        if not isinstance(step, int):
            traceback_and_raise(TypeError("expected integer type arguments for step"))
        else:
            step = step
        if self.start is None:
            start = length - 1 if step < 0 else 0
        if not isinstance(start, int):
            traceback_and_raise(TypeError("expected integer type arguments for start"))
        else:
            start = start
            if start < 0:
                start = start + length
        if self.stop is None:
            stop = -1 if step < 0 else length
        if not isinstance(stop, int):
            traceback_and_raise(TypeError("expected integer type arguments for stop"))
        else:
            stop = stop
            if stop < 0:
                stop = stop + length
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
        res = super().getindices(length, start, stop, step)
        return PrimitiveFactory.generate_primitive(value=res)

    def unpack(self, start: Any, stop: Any, step: Any) -> SyPrimitiveRet:
        if self.step is None:
            step = 1
        if step == 0:
            traceback_and_raise(TypeError("slice step can't be zero"))
        if step < -sys.maxsize:
            step = -sys.maxsize
        if self.start is None:
            start = sys.maxsize if step < 0 else 0

        if self.stop is None:
            stop = -sys.maxsize - 1 if step < 0 else sys.maxsize

        res = super().unpack(start, stop, step)
        return PrimitiveFactory.generate_primitive(value=res)

    def adjustindices(
        self, length: Any, start: Any, stop: Any, step: Any
    ) -> SyPrimitiveRet:
        assert step != 0
        assert step >= -sys.maxsize

        if start < 0:
            start = start + length
            if start < 0:
                start = -1 if step < 0 else 0
        else:
            if start >= length:
                start = length - 1 if step < 0 else length

        if stop < 0:
            stop = stop + length
            if stop < 0:
                stop = -1 if step < 0 else 0

        else:
            if stop >= length:
                stop = length - 1 if step < 0 else length

        if step < 0:
            if stop < start:
                return (start - stop - 1) / (-step) + 1

        else:
            if start < stop:
                return (stop - start - 1) / step + 1

        res = super().adjustindices(length, start, stop, step)
        return PrimitiveFactory.generate_primitive(value=res)

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
