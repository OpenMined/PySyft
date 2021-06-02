# stdlib
import functools
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# third party
import gym

# syft relative
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from ..util import generic_update_ast

LIB_NAME = "gym"
PACKAGE_SUPPORT = {
    "lib": LIB_NAME,
}


def create_ast(client: TypeAny = None) -> Globals:
    ast = Globals(client)

    modules: TypeList[TypeTuple[str, TypeAny]] = [
        ("gym", gym),
        ("gym.wrappers", gym.wrappers),
        ("gym.wrappers.time_limit", gym.wrappers.time_limit),
    ]
    classes: TypeList[TypeTuple[str, str, TypeAny]] = [
        (
            "gym.wrappers.time_limit.TimeLimit",
            "gym.wrappers.time_limit.TimeLimit",
            gym.wrappers.time_limit.TimeLimit,
        ),
        (
            "gym.Wrapper",
            "gym.Wrapper",
            gym.Wrapper,
        ),
    ]

    methods = [
        ("gym.make", "gym.wrappers.time_limit.TimeLimit"),
        ("gym.wrappers.time_limit.TimeLimit.seed", "syft.lib.python.List"),
        ("gym.wrappers.time_limit.TimeLimit.reset", "numpy.ndarray"),
        ("gym.wrappers.time_limit.TimeLimit.step", "syft.lib.python.Tuple"),
    ]

    add_modules(ast, modules)
    add_classes(ast, classes)
    add_methods(ast, methods)

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_storable_object_attr_convenience_methods()

    return ast


update_ast = functools.partial(generic_update_ast, LIB_NAME, create_ast)
