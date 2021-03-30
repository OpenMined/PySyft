# -*- coding: utf-8 -*-
"""This is a skeleton file that can serve as a starting point for Python console script.
To run this script uncomment the following lines in [options.entry_points] section in setup.cfg.
console_scripts =
     fibonacci = syft.skeleton:run
Then run `python setup.py install` which will install the command `fibonacci` inside your current environment.
Besides console scripts, header (i.e. until _logger...) of this file can also be used as template for Python modules.
Note: This skeleton file can be safely removed if not needed!
"""
# stdlib
from typing import List as TypeList

# third party
import torch

module_type = type(torch)
func_type = type(lambda x: x)
builtin_func_type = type(torch.ones)
class_type = type(func_type)


def unsplit(list_of_things: TypeList[str], separator: str = ".") -> str:
    """The unsplit method joins different parts of a node path using dot separator."""
    return separator.join(list_of_things)
