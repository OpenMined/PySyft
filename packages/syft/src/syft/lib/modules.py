# stdlib
# import sklearn
import functools

# import syft
import importlib
import sys
from typing import Any as TypeAny
from typing import Callable
from typing import Dict as TypeDict
from typing import Iterable
from typing import List as TypeList
from typing import Optional
from typing import Tuple as TypeTuple
from typing import Union as TypeUnion

# third party
from lib import create_lib_ast
from lib import load

# import sklearn
import wrapt

# syft relative
from ..ast import add_classes
from ..ast import add_methods
from ..ast import add_modules
from ..ast.globals import Globals
from .util import generic_update_ast

lib_ast = create_lib_ast(None)
bind_lib = ""


def register_library(lib: str, update_ast: Callable, objects):
    load(lib, ignore_warning=True)
    if "__SYFT_PACKAGE_SUPPORT" not in sys.modules:
        sys.modules["__SYFT_PACKAGE_SUPPORT"] = []
    sys.modules["__SYFT_PACKAGE_SUPPORT"].append(lib)
    bind_library(lib)


def bind_library(lib: str):
    global bind_lib
    bind_lib = f"sy.{lib}"
    package = "syft.lib"
    module_path = f"{package}.{lib}"
    globals()[bind_lib] = sys.modules[module_path]
