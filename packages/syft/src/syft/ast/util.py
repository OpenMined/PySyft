# -*- coding: utf-8 -*-
"""This module contains utility funtions for Syft's AST submodule."""

# stdlib
from typing import List as TypeList

# third party
import torch

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
