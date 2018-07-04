"""Framework agnostic static utility functions."""
import functools


def map_tuple(hook, args, func):
    if hook:
        return tuple(func(hook, x) for x in args)
    else:
        return tuple(func(x) for x in args)


def map_dict(hook, kwargs, func):
    if hook:
        return {key: func(hook, val) for key, val in kwargs.items()}
    else:
        return {key: func(val) for key, val in kwargs.items()}


def pass_method_args(method):
    """Wrapper gathering partialmethod object from method call."""
    @functools.wraps(method)
    def pass_args(*args, **kwargs):
        return functools.partialmethod(method, *args, **kwargs)
    return pass_args


def pass_func_args(func):
    """Wrapper gathering partial object from function call."""

    @functools.wraps(func)
    def pass_args(*args, **kwargs):
        # Return a new partial object which when called will behave like func called with the
        # positional arguments args and keyword arguments keywords. If more arguments are
        # supplied to the call, they are appended to args. If additional keyword arguments
        # are supplied, they extend and override keywords.

        # The partial() is used for partial function application which “freezes” some
        # portion of a function’s arguments and/or keywords resulting in a new object
        # with a simplified signature.
        return functools.partial(func, *args, **kwargs)
    return pass_args
