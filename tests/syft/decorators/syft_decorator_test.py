import inspect

import pytest
from syft.decorators import syft_decorator


def test_static():
    class Test:
        @staticmethod
        @syft_decorator()
        def test_static(x: int, y: int) -> None:
            pass

    Test.test_static(x=0, y=0)
    pass


def test_typecheck():
    @syft_decorator(typechecking=True)
    def test(x: int, y: int) -> int:
        return x + y

    test(x=1, y=2)

    with pytest.raises(TypeError) as e:
        test(x="test", y=2)

    assert str(e.value) == 'type of argument "x" must be int; got str instead'
    pass


def test_policy():
    @syft_decorator(enforce_policies=True)
    def test_policy_decorator() -> None:
        # TODO
        pass

    pass


def test_logger():
    @syft_decorator(syft_logger=True)
    def test_logger_decorator() -> None:
        # TODO
        pass

    pass


def test_compose_1():
    @syft_decorator(typechecking=True, syft_logger=True)
    def test() -> None:
        # TODO
        pass

    pass


def test_compose_2():
    @syft_decorator(typechecking=True, enforce_policies=True)
    def test() -> None:
        # TODO
        pass

    pass


def test_compose_3():
    @syft_decorator(typechecking=True, enforce_policies=True, syft_logger=True)
    def test() -> None:
        # TODO
        pass

    pass


def test_other():
    def decorator_1(func):
        def wrap(*arg, **kwargs):
            return func(*arg, **kwargs) + " decorator_1"

        return wrap

    def decorator_2(func):
        def wrap(*arg, **kwargs):
            return func(*arg, **kwargs) + " decorator_2"

        return wrap

    def decorator_3(func):
        def wrap(*arg, **kwargs):
            return func(*arg, **kwargs) + " decorator_3"

        return wrap

    @syft_decorator(other_decorators=[decorator_1, decorator_2, decorator_3])
    def decorated():
        return "func"

    assert decorated() == "func decorator_1 decorator_2 decorator_3"


def test_decorator_metadata():
    def decorator_1(func):
        """
        poorly written decorator, this does not copy the module, qualname, etc.
        """

        def wrap(*arg, **kwargs):
            return func(*arg, **kwargs) + 1

        return wrap

    def decorator_2(func):
        """
        poorly written decorator, this does not copy the module, qualname, etc.
        """

        def wrap(*arg, **kwargs):
            return func(*arg, **kwargs) + 2

        return wrap

    def fn(x: int, y: int) -> int:
        """
        Usually, bad written decorators can break documentation, typing, names or qualnames, we
        would like to not do that. This is some dummy documentation to test that.
        """
        return 0

    original_obj = fn

    @syft_decorator(typechecking=True, other_decorators=[decorator_1, decorator_2])
    def fn(x: int, y: int) -> int:
        """
        Usually, badly written decorators can break documentation, typing, names or qualnames, we
        would like to not do that. This is some dummy documentation to test that.
        """
        return 0

    decorated_obj = fn

    assert original_obj.__name__ == decorated_obj.__name__
    assert original_obj.__qualname__ == decorated_obj.__qualname__
    assert original_obj.__module__ == decorated_obj.__module__
    assert original_obj.__annotations__ == decorated_obj.__annotations__
    assert inspect.signature(original_obj) == inspect.signature(decorated_obj)
