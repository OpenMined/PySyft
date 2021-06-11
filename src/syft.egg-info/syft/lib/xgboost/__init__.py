"""Partial-dependency library, runs when user loads xgboost.

__init__ file for sklearn. This defines various modules, classes and methods which we currently support.
We create an AST for all these modules, classes and methods so that they can be called remotely.
"""
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
    """Create ast for all mdules, classes and attributes for sklearn so that they can be called remotely.

    Args:
        client: Remote client where we have to create the ast.

    Returns:
        Globals: returns ast created for xgboost.
    """
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
        # classifiers
        ("xgboost.XGBClassifier", "xgboost.XGBClassifier", xgb.XGBClassifier),
        ("xgboost.XGBRFClassifier", "xgboost.XGBRFClassifier", xgb.XGBRFClassifier),
        # ("xgboost.dask.DaskXGBRFClassifier"), Currently dask is not supported in syft
        # regreessors
        ("xgboost.XGBRegressor", "xgboost.XGBRegressor", xgb.XGBRegressor),
        ("xgboost.XGBRFRegressor", "xgboost.XGBRFRegressor", xgb.XGBRFRegressor),
        # ("xgboost.dask.DaskXGBRFRegressor"), Currently dask is not supported in syft
    ]

    methods = [
        ("xgboost.train", "xgboost.core.Booster"),
        ("xgboost.core.Booster.predict", "numpy.ndarray"),
        # classifiers
        ("xgboost.XGBClassifier.fit", "xgboost.XGBClassifier"),
        ("xgboost.XGBClassifier.predict", "numpy.ndarray"),
        ("xgboost.XGBRFClassifier.fit", "xgboost.XGBRFClassifier"),
        ("xgboost.XGBRFClassifier.predict", "numpy.ndarray"),
        # regressors
        ("xgboost.XGBRegressor.fit", "xgboost.XGBRegressor"),
        ("xgboost.XGBRegressor.predict", "numpy.ndarray"),
        ("xgboost.XGBRFRegressor.fit", "xgboost.XGBRFClassifier"),
        ("xgboost.XGBRFRegressor.predict", "numpy.ndarray"),
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
