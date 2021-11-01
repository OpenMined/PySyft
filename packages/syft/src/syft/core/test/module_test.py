# stdlib
from enum import Enum
from typing import Any

# relative
from ..common.serde.serializable import serializable

global_value: int = 5


def global_function() -> int:
    return global_value


@serializable(recursive_serde=True)
class A:
    """This is a test class to be used in functionality test."""

    __slots__ = ["_private_attr", "n"]
    __attr_allowlist__ = ["_private_attr", "n"]

    static_attr: int = 4

    def __init__(self) -> None:
        self._private_attr: float = 5.5

    def __len__(self) -> int:
        return 3

    def __iter__(self) -> "A":
        self.n = 1
        return self

    def __next__(self) -> int:
        if self.n < A.static_attr:
            result = self.n
            self.n += 1
            return result
        else:
            raise StopIteration

    def test_method(self) -> int:
        return 0

    @property
    def test_property(self) -> float:
        return self._private_attr

    @test_property.setter
    def test_property(self, value: Any) -> None:
        self._private_attr = value

    @staticmethod
    def static_method() -> bool:
        return True


class B(Enum):
    """This is a test enum to be used in functionality test."""

    Car = 1
    Cat = 2
    Dog = 3


@serializable(recursive_serde=True)
class IterWithoutLen:
    """This is a test class for testing iterator method in Klass."""

    __slots__ = ["n"]
    __attr_allowlist__ = ["n"]

    static_attr: int = 1

    def __iter__(self) -> "IterWithoutLen":
        self.n = 0
        return self

    def __next__(self) -> int:
        if self.n < A.static_attr:
            result = self.n
            self.n += 1
            return result
        else:
            raise StopIteration


@serializable(recursive_serde=True)
class C:

    __attr_allowlist__ = ["dynamic_object"]

    def __init__(self) -> None:
        self.dynamic_object = 123

    def dummy_reloadable_func(self) -> int:
        return 0

    @staticmethod
    def type_reload_func() -> None:
        C.dummy_reloadable_func = C.func_1  # type: ignore

    def obj_reload_func(self) -> None:
        print("RELOADING OBJ FUNCTION")
        self.dummy_reloadable_func = self.func_2  # type: ignore

    def func_1(self) -> int:
        return 1

    def func_2(self) -> int:
        return 2
