# stdlib
from typing import Any

# a dummy enum


class Meta(type):
    # any property returns another dummy which can also be executed
    def __getattribute__(cls, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except Exception:  # nosec
            pass
        return return_dummy()


# this lets us prevent runtime errors of missing types in older syft
class DummyNum(metaclass=Meta):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self


def return_dummy() -> DummyNum:
    # this lets us create the sub class in the parent meta on getattr
    return DummyNum()
