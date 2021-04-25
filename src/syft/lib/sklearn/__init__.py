"""Partial-dependency library, runs when user loads sklearn.

__init__ file for sklearn. This defines various modules, classes and methods which we currently support.
We create an AST for all these modules, classes and methods so that they can be called remotely.
"""

# stdlib
import functools
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# third party
import sklearn
import sklearn.linear_model

# syft relative
from . import serializing_models  # noqa: 401
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from ..util import generic_update_ast

LIB_NAME = "sklearn"
PACKAGE_SUPPORT = {"lib": LIB_NAME}


def create_ast(client: TypeAny = None) -> Globals:
    """Create ast for all mdules, classes and attributes for sklearn so that they can be called remotely.

    Args:
        client: Remote client where we have to create the ast.

    Returns:
        ast created for sklearn.
    """
    ast = Globals(client)

    modules: TypeList[TypeTuple[str, TypeAny]] = [
        ("sklearn", sklearn),
        ("sklearn.linear_model", sklearn.linear_model),
    ]

    classes: TypeList[TypeTuple[str, str, TypeAny]] = [
        ("sklearn.base", "sklearn.base", sklearn.base),
        # linear_model
        # LogisticRegression
        (
            "sklearn.linear_model.LogisticRegression",
            "sklearn.linear_model.LogisticRegression",
            sklearn.linear_model.LogisticRegression,
        ),
        # LinearRegression
        (
            "sklearn.linear_model.LinearRegression",
            "sklearn.linear_model._base.LinearRegression",
            sklearn.linear_model._base.LinearRegression,
        ),
    ]

    methods: TypeList[TypeTuple[str, str]] = [
        # linear_model
        # LogisticRegression
        (
            "sklearn.linear_model.LogisticRegression.fit",
            "sklearn.linear_model.LogisticRegression",
        ),
        ("sklearn.linear_model.LogisticRegression.predict", "pandas.DataFrame"),
        # LinearRegression
        (
            "sklearn.linear_model.LinearRegression.fit",
            "sklearn.linear_model._base.LinearRegression",
        ),
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
