import sys
import syft
import importlib
import sklearn

# import sklearn
import functools
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple
from typing import Dict as TypeDict
from typing import Union as TypeUnion
from typing import Callable
from typing import Iterable
from typing import Optional

from ..ast import add_classes
from ..ast import add_methods
from ..ast import add_modules
from ..ast.globals import Globals
from .util import generic_update_ast


def register_library(lib: str, update_ast: Callable, objects):

    if "_SYFT_PACKAGE_SUPPORT" not in sys.modules:
        sys.modules["_SYFT_PACKAGE_SUPPORT"] = []
    sys.modules["_SYFT_PACKAGE_SUPPORT"].append()
