# stdlib
from collections.abc import Iterable
from typing import Any
from typing import Generic
from typing import TypeVar

# relative
from ..service.action.action_object import ActionObject
from .uid import UID

T = TypeVar("T")
E = TypeVar("E")


class Ok(Generic[T]):
    def __init__(self, value: T):
        self.value = value


class Err(Generic[E]):
    def __init__(self, error: E):
        self.error = error


Result = Ok[T] | Err[E]

error_allowed_attrs = ["__class__"]
skip_attrs = ["_internal_names_set"]


class SyftAutoboxMeta(type):
    def __call__(cls, value: Result[T, E] | T) -> Any:
        if isinstance(value, Ok):
            wrapped_value = value.value
            is_error = False
        elif isinstance(value, Err):
            wrapped_value = value.error
            is_error = True
        else:
            wrapped_value = value
            is_error = False

        uid = None
        if isinstance(value, ActionObject):
            wrapped_value = value.syft_action_data
            uid = value.id

        wrapped_type = type(wrapped_value)
        dynamic_class_name = f"SyftAutobox[{wrapped_type.__name__}]"

        # Dynamically create a new class that inherits from the wrapped type only
        DynamicWrapper = type(
            f"SyftAutobox{wrapped_type.__name__.capitalize()}", (wrapped_type,), {}
        )

        class Wrapped(DynamicWrapper):  # type: ignore
            def __init__(self, value: Any, uid: UID | None = None):
                self._syft_value = value
                self._syft_uid = uid if uid is not None else UID()
                self._syft_is_error = is_error
                super(DynamicWrapper, self).__init__()

            def __getattribute__(self, name: str) -> Any:
                # Bypass certain attrs to prevent recursion issues
                if name.startswith("_syft") or name in skip_attrs:
                    return object.__getattribute__(self, name)

                if self._syft_is_error and name not in error_allowed_attrs:
                    raise Exception(
                        f"Cannot access attribute '{name}' on an Err result: {self._syft_value}"
                    )

                return getattr(self._syft_value, name)

            def __setattr__(self, name: str, value: Any) -> None:
                if name.startswith("_syft"):
                    object.__setattr__(self, name, value)
                    return
                setattr(self._syft_value, name, value)

            # these empty dunders are required
            def __repr__(self) -> str:
                return self.__repr__()

            def __str__(self) -> str:
                return self.__str__()

            def __dir__(self) -> Iterable[str]:
                return self.__dir__()

            def __getitem__(self, name: Any) -> Any:
                return self.__getitem__(name)

            def __setitem__(self, name: Any, value: Any) -> None:
                return self.__setitem__(name, value)

            def __array__(self) -> Any:
                return self.__array__()

            def __len__(self) -> int:
                return len(self._syft_value)

            def __iter__(self) -> Iterable:
                return iter(self._syft_value)

            def __contains__(self, item: Any) -> bool:
                return item in self._syft_value

            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                return self(*args, **kwargs)

            def __eq__(self, other: Any) -> bool:
                return self == other

            def __ne__(self, other: Any) -> bool:
                return self != other

            def __lt__(self, other: Any) -> bool:
                return self < other

            def __le__(self, other: Any) -> bool:
                return self <= other

            def __gt__(self, other: Any) -> bool:
                return self > other

            def __ge__(self, other: Any) -> bool:
                return self >= other

            def __add__(self, other: Any) -> Any:
                return self + other

            def __sub__(self, other: Any) -> Any:
                return self - other

            def __mul__(self, other: Any) -> Any:
                return self * other

            def __truediv__(self, other: Any) -> Any:
                return self / other

            def __floordiv__(self, other: Any) -> Any:
                return self // other

            def __mod__(self, other: Any) -> Any:
                return self % other

            def __pow__(self, other: Any, modulo: int | None = None) -> Any:
                return pow(self._syft_value, other, modulo)

            def __radd__(self, other: Any) -> Any:
                return other + self._syft_value

            def __rsub__(self, other: Any) -> Any:
                return other - self._syft_value

            def __rmul__(self, other: Any) -> Any:
                return other * self._syft_value

            def __rtruediv__(self, other: Any) -> Any:
                return other / self._syft_value

            def __rfloordiv__(self, other: Any) -> Any:
                return other // self._syft_value

            def __rmod__(self, other: Any) -> Any:
                return other % self._syft_value

            def __rpow__(self, other: Any) -> Any:
                return pow(other, self._syft_value)

            def __neg__(self) -> Any:
                return -self._syft_value

            def __pos__(self) -> Any:
                return +self._syft_value

            def __abs__(self) -> Any:
                return abs(self._syft_value)

            def __invert__(self) -> Any:
                return ~self._syft_value

            def __and__(self, other: Any) -> Any:
                return self & other

            def __or__(self, other: Any) -> Any:
                return self | other

            def __xor__(self, other: Any) -> Any:
                return self ^ other

            def __lshift__(self, other: Any) -> Any:
                return self << other

            def __rshift__(self, other: Any) -> Any:
                return self >> other

            def __rand__(self, other: Any) -> Any:
                return other & self

            def __ror__(self, other: Any) -> Any:
                return other | self

            def __rxor__(self, other: Any) -> Any:
                return other ^ self

            def __rlshift__(self, other: Any) -> Any:
                return other << self

            def __rrshift__(self, other: Any) -> Any:
                return other >> self

        Wrapped.__name__ = dynamic_class_name
        Wrapped.__qualname__ = dynamic_class_name
        obj = Wrapped(wrapped_value)
        if uid:
            obj._syft_uid = uid
        return obj


class SyftAutobox(Generic[T], metaclass=SyftAutoboxMeta):
    pass


Box = SyftAutobox
