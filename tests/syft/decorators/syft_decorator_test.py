# stdlib
import inspect
from typing import Any
from typing import Callable
from typing import Tuple

# third party
import pytest

# syft absolute
from syft.decorators import syft_decorator


def test_static() -> None:
    class Test:
        @staticmethod
        @syft_decorator()
        def test_static(x: int, y: int) -> None:
            pass

    Test.test_static(x=0, y=0)
    pass


def test_typecheck() -> None:
    @syft_decorator(typechecking=True)
    def test(x: int, y: int) -> int:
        return x + y

    test(x=1, y=2)

    with pytest.raises(TypeError) as e:
        test(x="test", y=2)

    assert str(e.value) == 'type of argument "x" must be int; got str instead'
    pass


def test_policy() -> None:
    @syft_decorator(enforce_policies=True)
    def test_policy_decorator() -> None:
        # TODO
        pass

    pass


def test_logger() -> None:
    @syft_decorator(syft_logger=True)
    def test_logger_decorator() -> None:
        # TODO
        pass

    pass


def test_compose_1() -> None:
    @syft_decorator(typechecking=True, syft_logger=True)
    def test() -> None:
        # TODO
        pass

    pass


def test_compose_2() -> None:
    @syft_decorator(typechecking=True, enforce_policies=True)
    def test() -> None:
        # TODO
        pass

    pass


def test_compose_3() -> None:
    @syft_decorator(typechecking=True, enforce_policies=True, syft_logger=True)
    def test() -> None:
        # TODO
        pass

    pass


def test_other() -> None:
    def decorator_1(func: Callable) -> Callable:
        def wrap(*arg: Tuple[Any, ...], **kwargs: Any) -> str:
            return func(*arg, **kwargs) + " decorator_1"

        return wrap

    def decorator_2(func: Callable) -> Callable:
        def wrap(*arg: Tuple[Any, ...], **kwargs: Any) -> str:
            return func(*arg, **kwargs) + " decorator_2"

        return wrap

    def decorator_3(func: Callable) -> Callable:
        def wrap(*arg: Tuple[Any, ...], **kwargs: Any) -> str:
            return func(*arg, **kwargs) + " decorator_3"

        return wrap

    @syft_decorator(other_decorators=[decorator_1, decorator_2, decorator_3])
    def decorated() -> str:
        return "func"

    assert decorated() == "func decorator_1 decorator_2 decorator_3"


def test_decorator_metadata() -> None:
    def decorator_1(func: Callable) -> Callable:
        """
        poorly written decorator, this does not copy the module, qualname, etc.
        """

        def wrap(*arg: Tuple[Any, ...], **kwargs: Any) -> int:
            return func(*arg, **kwargs) + 1

        return wrap

    def decorator_2(func: Callable) -> Callable:
        """
        poorly written decorator, this does not copy the module, qualname, etc.
        """

        def wrap(*arg: Tuple[Any, ...], **kwargs: Any) -> int:
            return func(*arg, **kwargs) + 2

        return wrap

    def fn(x: int, y: int) -> int:
        """
        Usually, badly written decorators can break documentation, typing, names or qualnames, we
        would like to not do that. This is some dummy documentation to test that.
        """
        return 0

    original_obj = fn

    @syft_decorator(typechecking=True, other_decorators=[decorator_1, decorator_2])
    def fn2(x: int, y: int) -> int:
        """
        Usually, badly written decorators can break documentation, typing, names or qualnames, we
        would like to not do that. This is some dummy documentation to test that.
        """
        return 0

    decorated_obj = fn2

    # fn cant be redeclared so it should be named fn2
    assert original_obj.__name__ + "2" == decorated_obj.__name__
    assert original_obj.__qualname__ + "2" == decorated_obj.__qualname__
    assert original_obj.__module__ == decorated_obj.__module__
    assert original_obj.__annotations__ == decorated_obj.__annotations__
    assert inspect.signature(original_obj) == inspect.signature(decorated_obj)
