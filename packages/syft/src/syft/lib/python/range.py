# stdlib
from typing import Any
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# relative
from ...core.common.serde.serializable import serializable
from ...proto.lib.python.range_pb2 import Range as Range_PB
from .iterator import Iterator
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive
from .types import SyPrimitiveRet


@serializable()
class Range(PyPrimitive):
    __slots__ = ["_index"]

    def __init__(
        self,
        start: Any = None,
        stop: Any = None,
        step: Any = 1,
    ):
        if stop is None:
            stop = start
            start = 0
        self.value = range(start, stop, step)

    def __contains__(self, other: Any) -> SyPrimitiveRet:
        res = self.value.__contains__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __eq__(self, other: Any) -> SyPrimitiveRet:
        res = self.value.__eq__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __ne__(self, other: Any) -> SyPrimitiveRet:
        res = self.value.__ne__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __sizeof__(self) -> SyPrimitiveRet:
        res = self.value.__sizeof__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __bool__(self) -> SyPrimitiveRet:
        # res = self.value.__bool__()
        # mypy error: "range" has no attribute "__bool__"
        # work around:
        try:
            res = bool(self.value.__len__())
        except OverflowError:
            res = True
        return PrimitiveFactory.generate_primitive(value=res)

    def __len__(self) -> Any:
        res = self.value.__len__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __getitem__(self, key: int) -> Any:
        res = self.value.__getitem__(key)
        return PrimitiveFactory.generate_primitive(value=res)

    def __iter__(self, max_len: Optional[int] = None) -> Iterator:
        return Iterator(self.value, max_len=max_len)

    @property
    def start(self) -> SyPrimitiveRet:
        res = self.value.start
        return PrimitiveFactory.generate_primitive(value=res)

    @property
    def step(self) -> SyPrimitiveRet:
        res = self.value.step
        return PrimitiveFactory.generate_primitive(value=res)

    @property
    def stop(self) -> SyPrimitiveRet:
        res = self.value.stop
        return PrimitiveFactory.generate_primitive(value=res)

    def index(self, value: int) -> SyPrimitiveRet:
        res = self.value.index(value)
        return PrimitiveFactory.generate_primitive(value=res)

    def count(self, value: int) -> SyPrimitiveRet:
        res = self.value.count(value)
        return PrimitiveFactory.generate_primitive(value=res)

    def upcast(self) -> range:
        return self.value

    def _object2proto(self) -> Range_PB:
        range_pb = Range_PB()

        range_pb.start = self.start
        range_pb.stop = self.stop
        range_pb.step = self.step

        return range_pb

    @staticmethod
    def _proto2object(proto: Range_PB) -> "Range":

        return Range(
            start=proto.start,
            stop=proto.stop,
            step=proto.step,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Range_PB
