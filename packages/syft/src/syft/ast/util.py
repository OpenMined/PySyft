# -*- coding: utf-8 -*-
"""This module contains utility funtions for Syft's AST submodule."""

# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import List as TypeList
from typing import Tuple
from typing import Union

# third party
import torch

# relative
from ..core.node.common.action.action_sequence import ActionSequence
from ..core.pointer.pointer import Pointer

module_type = type(torch)
func_type = type(lambda x: x)
builtin_func_type = type(torch.ones)
class_type = type(func_type)


def unsplit(list_of_things: TypeList[str], separator: str = ".") -> str:
    """The unsplit method joins different parts of a node path using dot separator.

    Args:
        list_of_things: list of string to join.
        separator: Separator character/string. (default ".")

    Returns:
        str: string after join.
    """
    return separator.join(list_of_things)


# TODO: this should move out of AST into a util somewhere? or osmething related to Pointer
def pointerize_args_and_kwargs(
    args: Union[List[Any], Tuple[Any, ...]],
    kwargs: Dict[Any, Any],
    client: Any,
    gc_enabled: bool = True,
) -> Tuple[List[Any], Dict[Any, Any]]:
    """Get pointers to args and kwargs.

    Args:
        args: List of arguments.
        kwargs: Dict of Keyword arguments.
        client: Client node.

    Returns:
        Tuple of args and kwargs with pointer to values.
    """
    # When we try to send params to a remote function they need to be pointers so
    # that they can be serialized and fetched from the remote store on arrival
    # this ensures that any args which are passed in from the user side are first
    # converted to pointers and sent then the pointer values are used for the
    # method invocation
    obj_lst = []
    pointer_args = []
    pointer_kwargs = {}
    for arg in args:
        # check if its already a pointer
        if not isinstance(arg, Pointer):
            arg_ptr, obj = arg.send(client, pointable=not gc_enabled, immediate=False)
            obj_lst.append(obj)
            pointer_args.append(arg_ptr)
        else:
            pointer_args.append(arg)
            arg.gc_enabled = gc_enabled

    for k, arg in kwargs.items():
        # check if its already a pointer
        if not isinstance(arg, Pointer):
            arg_ptr, obj = arg.send(client, pointable=not gc_enabled, immediate=False)
            obj_lst.append(obj)
            pointer_kwargs[k] = arg_ptr
        else:
            pointer_kwargs[k] = arg

    if obj_lst:
        msg = ActionSequence(obj_lst=obj_lst, address=client.address)

        # send message to client
        client.send_immediate_msg_without_reply(msg=msg)

    return pointer_args, pointer_kwargs
