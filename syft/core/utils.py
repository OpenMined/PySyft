"""Framework agnostic static utility functions."""
import functools
import json
import logging
import re
import types
from typing import Optional, Callable, Tuple

import torch

from ._types import PYTHON_ENCODE_RETURN_TYPE, CUSTOM_OBJECT_HOOK_RETURN_TYPE, Dict, Any


is_in_place_method_dict = {}


def is_in_place_method(attr: str) -> bool:
    """
    Determines if the method is in-place (ie modifies the self)
    TODO: Can you do better?
    """
    try:
        return is_in_place_method_dict[attr]
    except KeyError:
        pat = re.compile('__(.+)__')
        is_in_place = pat.search(attr) is None and attr[-1] == '_'
        is_in_place_method_dict[attr] = is_in_place
        return is_in_place

def map_tuple(hook, args: Tuple[Any, ...], func: Callable) -> Tuple[Any, ...]:
    if hook:
        return tuple(func(hook, x) for x in args)
    else:
        return tuple(func(x) for x in args)


def map_dict(hook, kwargs: Dict[str, Any], func: Callable) -> Dict[str, Any]:
    if hook:
        return {key: func(hook, val) for key, val in kwargs.items()}
    else:
        return {key: func(val) for key, val in kwargs.items()}


def pass_method_args(method: Callable) -> Callable:
    """Wrapper gathering partialmethod object from method call."""

    @functools.wraps(method)
    def pass_args(*args: Tuple[Any, ...], **kwargs: Dict[str, Any]):
        return functools.partialmethod(method, *args, **kwargs)

    return pass_args


def pass_func_args(func: Callable) -> Callable:
    """Wrapper gathering partial object from function call."""

    @functools.wraps(func)
    def pass_args(*args: Tuple[Any, ...], **kwargs: Dict[str, Any]):
        # Return a new partial object which when called will behave like func called with the
        # positional arguments args and keyword arguments keywords. If more arguments are
        # supplied to the call, they are appended to args. If additional keyword arguments
        # are supplied, they extend and override keywords.
        # The partial() is used for partial function application which "freezes" some
        # portion of a function's arguments and/or keywords resulting in a new object
        # with a simplified signature.
        return functools.partial(func, *args, **kwargs)

    return pass_args
