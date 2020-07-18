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
        """In this method, we want to check to see if all arguments (except self) are passed in as
        kwargs. Additionally, we want to have an informative error for when args are passed in
        incorrectly. The requirement to only use kwargs is a bit of an exotic one and some Python
        users might not be used to it. Thus, a good error message is important."""

        # We begin by initializing the maximum number of args we will allow at 0. We will iterate
        # this if by chance we see an argument whose name is "self".
        max_arg_len = 0

        # iterate through every parameter passed in
        for idx, param_name in enumerate(literal_signature.parameters):

            if idx == 0 and param_name == "self":
                max_arg_len += 1
                continue

            # if this parameter isn't in kwargs, then it's probably in args. However, we can't check
            # directly because we don't have arg names, only the list of args which were passed in.
            # Thus, the way this check works is to return an error if we find an argument which
            # isn't in kwargs and isn't "self".
            if param_name not in kwargs and len(args) > max_arg_len:

                raise AttributeError(
                    f"'{param_name}' was passed into a function as an arg instead of a kwarg. "
                    f"Please pass in all arguments as kwargs when coding/using PySyft."
                )

    if (
        literal_signature.return_annotation is literal_signature.empty
        and decorated.__name__ not in SKIP_RETURN_TYPE_HINTS
    ):
        raise AttributeError(
            f"Return type not annotated, please provide typing to the return type for function"
            f"{decorated.__qualname__}."
        )

    for idx, (param_name, param) in enumerate(literal_signature.parameters.items()):
        if idx == 0 and param_name == "self":
            continue

        if param_name not in solved_signature:
            raise AttributeError(
                f"Argument types not annotated, please provide typing to all argument types for"
                f"function {decorated.__qualname__}."
            )

    def decorator(*args, **kwargs):
        check_args(*args, **kwargs)
        return typechecked(decorated)(*args, **kwargs)

    decorator.__annotations__ = decorated.__annotations__
    decorator.__qualname__ = decorated.__qualname__
    decorator.__name__ = decorated.__name__
    decorator.__doc__ = decorated.__doc__
    decorator.__module__ = decorated.__module__

    old_signature = inspect.signature(decorated)
    decorator.__signature__ = old_signature

    return decorator
