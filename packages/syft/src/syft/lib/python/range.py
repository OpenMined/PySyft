# stdlib
from typing import Any
from typing import Optional
from typing import Union

# third party
from typing_extensions import SupportsIndex

# relative
from ...core.common import UID
from .iterator import Iterator
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive
from .types import SyPrimitiveRet


class Range(PyPrimitive):
    __slots__ = ["_id", "_index"]

    def __init__(
        self,
        start: Any = None,
        stop: Optional[Any] = None,
        step: SupportsIndex = 1,
        id: Optional[UID] = None,
    ):
        if stop is None:
            stop = start
            start = 0
        self.value = range(start, stop, step)
        self._id: UID = id if id else UID()

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

    def __getitem__(self, key: Union[int]) -> Any:
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
