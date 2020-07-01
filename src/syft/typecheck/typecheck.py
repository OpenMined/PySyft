import inspect
import typing
from typeguard import typechecked

SKIP_RETURN_TYPE_HINTS = {"__init__"}


def type_hints(decorated: typing.Callable) -> typing.Callable:
    """
    Decorator to enforce typechecking using the type hints of a function.

    If you use this decorator you have to:
    1. type your arguments and return type using explicit typing.
    2. address those arguments by their keys.

    Example:

    @type_hints
    def func(x: int, y: int) -> int:
    return x + y

    func(x = 1, y = 2)
    """

    literal_signature = inspect.signature(decorated)
    solved_signature = typing.get_type_hints(decorated)

    def check_args(*args, **kwargs):
        for idx, param_name in enumerate(literal_signature.parameters):
            if idx == 0 and param_name == "self":
                continue

            if not param_name in kwargs:
                raise AttributeError(
                    f"'{param_name}' was passed into a function as an arg instead of a kwarg."
                    f"Please pass in arguments as kwargs."
                )

    if (
        literal_signature.return_annotation is literal_signature.empty
        and decorated.__name__ not in SKIP_RETURN_TYPE_HINTS
    ):
        raise AttributeError(
            f"Return type not annotated, please provide typing to the return type for function {decorated.__qualname__}."
        )

    for idx, (param_name, param) in enumerate(literal_signature.parameters.items()):
        if idx == 0 and param_name == "self":
            continue

        if param_name not in solved_signature:
            raise AttributeError(
                f"Argument types not annotated, please provide typing to all argument types for function {decorated.__qualname__}."
            )

    def decorator(*args, **kwargs):
        check_args(*args, **kwargs)
        return typechecked(decorated)(*args, **kwargs)

    decorator.__qualname__ = decorated.__qualname__
    return decorator
