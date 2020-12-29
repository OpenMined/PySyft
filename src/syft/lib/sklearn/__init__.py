# stdlib
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple
from typing import Union as TypeUnion

# third party
import sklearn
import sklearn.linear_model

# syft relative
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals

LIB_NAME = "sklearn"
PACKAGE_SUPPORT = {"lib": LIB_NAME}


# this gets called on global ast as well as clients
# anything which wants to have its ast updated and has an add_attr method
def update_ast(ast: TypeUnion[Globals, TypeAny]) -> None:
    sklearn_ast = create_ast()
    ast.add_attr(attr_name=LIB_NAME, attr=sklearn_ast.attrs[LIB_NAME])


def create_ast() -> Globals:
    ast = Globals()

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
        klass.create_serialization_methods()
        klass.create_storable_object_attr_convenience_methods()

    return ast
