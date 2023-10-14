# stdlib
from collections import OrderedDict
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from types import MappingProxyType
from typing import Generic
from typing import Optional
from typing import SupportsIndex
from typing import TypeVar
from typing import Union
from typing import overload

# third party
from typing_extensions import Self

_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class TupleDict(OrderedDict[_KT, _VT]):
    def __getitem__(self, key: Union[int, _KT]) -> _VT:
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)

    def __len__(self) -> int:
        return len(self.keys())

    def __iter__(self) -> Iterator[_VT]:
        yield from self.values()


class _Meta(type):
    def __call__(
        cls: type[_T],
        __value: Optional[Iterable] = None,
        __key: Optional[Collection] = None,
        /,
    ) -> _T:
        if __value is None and __key is None:
            obj = cls.__new__(cls)
            obj.__init__()
            return obj

        if isinstance(__value, Mapping) and __key is None:
            keys = OrderedDict()
            values = []

            for i, k in enumerate(__value.keys()):
                keys[k] = i
                values.append(__value[k])

            obj = cls.__new__(cls, values)
            obj.__init__(keys)

            return obj

        if isinstance(__value, Iterable) and __key is None:
            keys = OrderedDict()
            values = []

            for i, (k, v) in enumerate(__value):
                keys[k] = i
                values.append(v)

            obj = cls.__new__(cls, values)
            obj.__init__(keys)

            return obj

        if isinstance(__value, Iterable) and __key is not None:
            if len(__key) != len(__value):
                raise ValueError("`__keys` has to be of the same length as `__values`")

            keys = OrderedDict((k, i) for i, k in enumerate(__key))

            obj = cls.__new__(cls, __value)
            obj.__init__(keys)

            return obj

        raise NotImplementedError


class DictTuple(tuple[_VT, ...], Generic[_KT, _VT], metaclass=_Meta):
    __mapping: MappingProxyType[_KT, int]

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, __value: Iterable[tuple[_KT, _VT]]) -> None:
        ...

    @overload
    def __init__(self, __value: Mapping[_KT, _VT]) -> None:
        ...

    @overload
    def __init__(self, __value: Iterable[_VT], __key: Collection[_KT]) -> None:
        ...

    def __init__(
        self, __value: Optional[Union[Mapping[_KT, int], Iterable[_KT]]] = None, /
    ) -> None:
        match __value:
            case MappingProxyType():
                self.__mapping = __value
            case Mapping():
                self.__mapping = MappingProxyType(__value)
            case Iterable():
                self.__mapping = MappingProxyType(
                    OrderedDict((k, i) for i, k in enumerate(__value))
                )
            case _:
                pass

        super().__init__()

    @overload
    def __getitem__(self, __key: _KT) -> _VT:
        ...

    @overload
    def __getitem__(self, __key: slice) -> Self:
        ...

    @overload
    def __getitem__(self, __key: SupportsIndex) -> _VT:
        ...

    def __getitem__(self, __key):
        if isinstance(__key, slice):
            return self.__class__(
                super().__getitem__(__key),
                list(self.__mapping.keys()).__getitem__(__key),
            )

        if isinstance(__key, SupportsIndex):
            return super().__getitem__(__key)

        return super().__getitem__(self.__mapping[__key])

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}{super().__repr__()}"

    def keys(self):
        return self.__mapping.keys()
