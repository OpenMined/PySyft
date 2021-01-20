# stdlib
from enum import Enum
from typing import Any

global_value: int = 5


def global_function() -> int:
    return global_value


class A:
    __slots__ = ["_private_attr"]

    static_attr: int = 4

    def __init__(self) -> None:
        self._private_attr: float = 5.5

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
    Car = 1
    Cat = 2
    Dog = 3
