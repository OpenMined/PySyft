# stdlib
from abc import ABCMeta
from collections import OrderedDict
from collections import deque
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import KeysView
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any
from typing import Generic
from typing import SupportsIndex
from typing import TypeVar
from typing import get_args
from typing import get_origin
from typing import overload

# third party
from pydantic import GetCoreSchemaHandler
from pydantic import ValidatorFunctionWrapHandler
from pydantic_core import CoreSchema
from pydantic_core import core_schema
from typing_extensions import Self

_T = TypeVar("_T", bound="DictTuple")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


# To correctly implement the creation of a DictTuple instance with
# DictTuple(key_value_pairs: Iterable[tuple[_KT, _VT]]),
# implementing just __init__ and __new__ is not enough.
#
# We need to extract the keys and values from keys_value_pairs
# and pass the values to __new__ (needs to call tuple.__new__(values) to create the tuple)
# and the keys to __init__ to create the mapping _KT -> int,
# but we can only iterate over key_value_pairs once since keys_value_pairs,
# being just an Iterable, might be ephemeral.
#
# Implementing just __new__ and __init__ for DictTuple is not enough since when
# calling DictTuple(key_value_pairs), __new__(keys_and_values) and __init__(key_value_pairs)
# are called in 2 separate function calls. If keys_and_values are ephemeral, like a generator,
# by the time it's passed to __init__() it's already been exhausted and there is no way to
# extract the keys out to create the mapping.
#
# Thus it is necessary to override __call__ of the metaclass
# to customize the way __new__ and __init__ work together, by iterating over key_value_pairs
# once to extract both keys and values, then passing keys to __new__, values to __init__
# within the same function call.
class _Meta(ABCMeta):
    @overload
    def __call__(cls: type[_T], /) -> _T: ...  # type: ignore[misc]

    @overload
    def __call__(cls: type[_T], __value: Iterable[tuple[_KT, _VT]], /) -> _T: ...  # type: ignore[misc]

    @overload
    def __call__(cls: type[_T], __value: Mapping[_KT, _VT], /) -> _T: ...  # type: ignore[misc]

    @overload
    def __call__(  # type: ignore[misc]
        cls: type[_T], __value: Iterable[_VT], __key: Collection[_KT], /
    ) -> _T: ...

    @overload
    def __call__(  # type: ignore[misc]
        cls: type[_T], __value: Iterable[_VT], __key: Callable[[_VT], _KT], /
    ) -> _T: ...

    def __call__(  # type: ignore[misc]
        cls: type[_T],
        __value: Iterable | None = None,
        __key: Callable | Collection | None = None,
        /,
    ) -> _T:
        # DictTuple()
        if __value is None and __key is None:
            obj = cls.__new__(cls)
            obj.__init__()  # type: ignore[misc]
            return obj

        # DictTuple(DictTuple(...))
        elif type(__value) is cls:
            return __value

        # DictTuple({"x": 123, "y": 456})
        elif isinstance(__value, Mapping) and __key is None:
            obj = cls.__new__(cls, __value.values())
            obj.__init__(__value.keys())  # type: ignore[misc]

            return obj

        # DictTuple(EnhancedDictTuple(...))
        # EnhancedDictTuple(DictTuple(...))
        # where EnhancedDictTuple subclasses DictTuple
        elif callable(__value_items := getattr(__value, "items", None)):
            return cls.__call__(__value_items())

        # DictTuple([("x", 123), ("y", 456)])
        elif isinstance(__value, Iterable) and __key is None:
            keys = OrderedDict()
            values: deque = deque()

            for i, (k, v) in enumerate(__value):
                keys[k] = i
                values.append(v)

            obj = cls.__new__(cls, values)
            obj.__init__(keys)  # type: ignore[misc]

            return obj

        # DictTuple([123, 456], ["x", "y"])
        elif isinstance(__value, Iterable) and isinstance(__key, Iterable):
            keys = OrderedDict((k, i) for i, k in enumerate(__key))

            obj = cls.__new__(cls, __value)
            obj.__init__(keys)  # type: ignore[misc]

            return obj

        # DictTuple(["abc", "xyz"], lambda x: x[0])
        # equivalent to DictTuple({"a": "abc", "x": "xyz"})
        elif isinstance(__value, Iterable) and isinstance(__key, Callable):
            obj = cls.__new__(cls, __value)  # type: ignore[misc]
            obj.__init__(__key)  # type: ignore[misc]

            return obj

        raise NotImplementedError


class DictTuple(tuple[_VT, ...], Generic[_KT, _VT], metaclass=_Meta):
    """
    OVERVIEW

        tuple with support for dict-like __getitem__(key)

            dict_tuple = DictTuple({"x": 1, "y": 2})

            dict_tuple["x"] == 1

            dict_tuple["y"] == 2

            dict_tuple[0] == 1

            dict_tuple[1] == 2

        everything else, e.g. __contains__, __iter__, behaves similarly to a tuple


    CREATION

        DictTuple(iterable) -> DictTuple([("x", 1), ("y", 2)])

        DictTuple(mapping) -> DictTuple({"x": 1, "y": 2})

        DictTuple(values, keys) -> DictTuple([1, 2], ["x", "y"])


    IMPLEMENTATION DETAILS

        DictTuple[_KT, _VT] is essentially a tuple[_VT, ...] that maintains an immutable Mapping[_KT, int]
        from the key to the tuple index internally.

        For example DictTuple({"x": 12, "y": 34}) is just a tuple (12, 34) with a {"x": 0, "y": 1} mapping.

        types.MappingProxyType is used for the mapping for immutability.
    """

    __mapping: MappingProxyType[_KT, int]

    # These overloads are copied from _Meta.__call__ just for IDE hints
    @overload
    def __init__(self, /) -> None: ...

    @overload
    def __init__(self, __value: Iterable[tuple[_KT, _VT]], /) -> None: ...

    @overload
    def __init__(self, __value: Mapping[_KT, _VT], /) -> None: ...

    @overload
    def __init__(self, __value: Iterable[_VT], __key: Collection[_KT], /) -> None: ...

    @overload
    def __init__(
        self, __value: Iterable[_VT], __key: Callable[[_VT], _KT], /
    ) -> None: ...

    def __init__(self, __value: Any = None, /) -> None:  # type: ignore[misc]
        if isinstance(__value, MappingProxyType):
            self.__mapping = __value
        elif isinstance(__value, Mapping):
            self.__mapping = MappingProxyType(__value)
        elif isinstance(__value, Iterable):
            self.__mapping = MappingProxyType(
                OrderedDict((k, i) for i, k in enumerate(__value))
            )
        elif isinstance(__value, Callable):
            self.__mapping = MappingProxyType(
                OrderedDict((__value(v), i) for i, v in enumerate(self))
            )

        super().__init__()

        if len(self.__mapping) != len(self):
            raise ValueError("`__keys` and `__values` do not have the same length")

        if any(isinstance(k, SupportsIndex) for k in self.__mapping.keys()):
            raise ValueError(
                "values of `__keys` should not have type `int`, "
                "or implement `__index__()`"
            )

    @overload  # type: ignore[override]
    def __getitem__(self, __key: _KT, /) -> _VT: ...

    @overload  # type: ignore[overload-overlap]
    def __getitem__(self, __key: slice, /) -> Self: ...

    @overload
    def __getitem__(self, __key: SupportsIndex, /) -> _VT: ...

    def __getitem__(self, __key: _KT | slice | SupportsIndex, /) -> _VT | Self:
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

    def _repr_html_(self) -> str:
        super_repr_html = getattr(super(), "_repr_html_", None)
        if super_repr_html is None:
            return repr(self)
        if len(self) == 0:  # empty DictTuple
            return "0 records"
        return super_repr_html()

    def keys(self) -> KeysView[_KT]:
        return self.__mapping.keys()

    def items(self) -> Iterable[tuple[_KT, _VT]]:
        return zip(self.__mapping.keys(), self)

    # https://docs.pydantic.dev/latest/concepts/types/#handling-custom-generic-classes
    # pydantic validator
    # this enables annotating a field with DictTuple[K, V] instead of just DictTuple
    # inside a pydantic BaseModel, e.g.
    #
    # class DatasetPageView(BaseModel):
    #     datasets: DictTuple[str, Dataset]
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        origin = get_origin(source_type)
        if origin is None:
            origin = source_type
            kt, vt = (Any, Any)
        else:
            kt, vt, *_ = get_args(source_type)

        k_schema = handler.generate_schema(MappingProxyType[kt, int])
        v_schema = handler.generate_schema(vt)

        def val_k(v: cls, handler: ValidatorFunctionWrapHandler) -> cls:
            handler(v.__mapping)
            return v

        def val_v(v: cls, handler: ValidatorFunctionWrapHandler) -> cls:
            handler(v)
            return v

        # pydantic validator for DictTuple[K, V]
        # - check that object has type DictTuple
        # - check that object is a tuple[V]
        # - check that the keys have type K
        return core_schema.chain_schema(
            [
                core_schema.is_instance_schema(cls),
                core_schema.no_info_wrap_validator_function(
                    val_v, core_schema.tuple_variable_schema(items_schema=v_schema)
                ),
                core_schema.no_info_wrap_validator_function(val_k, k_schema),
            ]
        )
