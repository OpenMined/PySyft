# stdlib
from enum import Enum
from typing import Any

# third party
import pandas as pd

# syft absolute
from syft.core.common.serde.recursive import RecursiveSerde
from syft.generate_wrapper import GenerateAutoSerdeWrapper

global_value: int = 5


def global_function() -> int:
    return global_value


class A(RecursiveSerde):
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


class IterWithoutLen(RecursiveSerde):
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


class C(RecursiveSerde):

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


class D:
    def __init__(self, n: int, data: pd.DataFrame):
        self.n = n
        self.data = data

    def get_n(self) -> int:
        return self.n

    def get_data(self) -> pd.DataFrame:
        return self.data


class DWrapper(RecursiveSerde, D):
    __attr_allowlist__ = ["n", "data"]

    def __init__(self, value):
        """The wrapper recieves instance of class as value.

        Args:
            value: instance of `module_test.D`
        """
        self.obj = value
        # create instance of Wrapper based on value (instance of class)
        super().__init__(value.n, value.data)

    def upcast(self):
        return self.obj

    @staticmethod
    def wrapped_type() -> type:
        return D


GenerateAutoSerdeWrapper(DWrapper, D, "module_test.D")
