# stdlib
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple
from typing import Union as TypeUnion

# third party
import pandas as pd

# syft relative
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from .frame import PandasDataFrameWrapper  # noqa: 401

LIB_NAME = "pandas"
PACKAGE_SUPPORT = {"lib": LIB_NAME}


# this gets called on global ast as well as clients
# anything which wants to have its ast updated and has an add_attr method
def update_ast(ast: TypeUnion[Globals, TypeAny]) -> None:
    pandas_ast = create_ast()
    ast.add_attr(attr_name=LIB_NAME, attr=pandas_ast.attrs[LIB_NAME])


def create_ast() -> Globals:
    ast = Globals()

    modules: TypeList[TypeTuple[str, TypeAny]] = [("pandas", pd)]

    classes: TypeList[TypeTuple[str, str, TypeAny]] = [
        ("pandas.DataFrame", "pandas.core.frame.DataFrame", pd.DataFrame),
    ]

    methods: TypeList[TypeTuple[str, str]] = []

    add_modules(ast, modules)
    add_classes(ast, classes)
    add_methods(ast, methods)

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_serialization_methods()
        klass.create_storable_object_attr_convenience_methods()

    return ast
