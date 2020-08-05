import inspect

from .typecheck import type_hints

# this flag is set in syft.__init__.py
LONG_TYPECHECK_STACK_TRACES = None


def syft_decorator(
    typechecking=False,
    prohibit_args=True,
    enforce_policies=False,
    syft_logger=False,
    other_decorators: list = None,
):
    def decorator(function):

        if typechecking:

            function = type_hints(function, prohibit_args=prohibit_args)

        def wrapper(*args, **kwargs):

            return function(*args, **kwargs)
            # try:
            #     return function(*args, **kwargs)
            # except Exception as e:
            #     if LONG_TYPECHECK_STACK_TRACES:
            #         raise e
            #     # Truncate stacktrace concerned with the type checking decorator
            #     # so that the true problem is easier to see
            #     raise Exception(str(e))

        if other_decorators:
            for other_decorator in other_decorators:
                wrapper = other_decorator(wrapper)

        wrapper.__annotations__ = function.__annotations__
        wrapper.__qualname__ = function.__qualname__
        wrapper.__name__ = function.__name__
        wrapper.__doc__ = function.__doc__
        wrapper.__module__ = function.__module__

        old_signature = inspect.signature(function)
        wrapper.__signature__ = old_signature

        return wrapper

    return decorator
