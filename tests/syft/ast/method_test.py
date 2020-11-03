# syft absolute
from syft.ast.callable import Callable
from syft.ast.method import Method


def test_method_constructor() -> None:
    method = Method()
    assert issubclass(type(method), Callable)
