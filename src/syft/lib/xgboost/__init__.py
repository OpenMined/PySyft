# stdlib
import functools
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# third party
import xgboost as xgb

# syft relative
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from ..util import generic_update_ast

LIB_NAME = "xgboost"
PACKAGE_SUPPORT = {
    "lib": LIB_NAME,
}


def create_ast(client: TypeAny) -> Globals:
    ast = Globals(client=client)

    modules: TypeList[TypeTuple[str, TypeAny]] = [
        ("xgboost", xgb),
        ("xgboost.core", xgb.core),
        ("xgboost.sklearn", xgb.sklearn),
    ]

    classes: TypeList[TypeTuple[str, str, TypeAny]] = [
        ("xgboost.DMatrix", "xgboost.DMatrix", xgb.core.DMatrix),
        ("xgboost.core.DMatrix", "xgboost.core.DMatrix", xgb.core.DMatrix),
        ("xgboost.core.Booster", "xgboost.core.Booster", xgb.core.Booster),
        (
            "xgboost.core.XGBoostError",
            "xgboost.core.XGBoostError",
            xgb.core.XGBoostError,
        ),
        ("xgboost.XGBClassifier", "xgboost.XGBClassifier", xgb.XGBClassifier),
        ("xgboost.XGBRegressor", "xgboost.XGBRegressor", xgb.XGBRegressor),
    ]

    methods = [
        ("xgboost.train", "xgboost.core.Booster"),
        ("xgboost.core.Booster.predict", "numpy.ndarray"),
        ("xgboost.XGBClassifier.fit", "xgboost.XGBClassifier"),
        ("xgboost.XGBClassifier.predict", "numpy.ndarray"),
        ("xgboost.XGBRegressor.fit", "xgboost.XGBRegressor"),
        ("xgboost.XGBRegressor.predict", "numpy.ndarray"),
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
