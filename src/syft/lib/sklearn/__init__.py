# stdlib
import functools
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# third party
import sklearn
import sklearn.base
import sklearn.cluster
import sklearn.compose
import sklearn.covariance
import sklearn.cross_decomposition
import sklearn.decomposition
import sklearn.discriminant_analysis
import sklearn.dummy
import sklearn.ensemble
import sklearn.kernel_ridge
import sklearn.linear_model
import sklearn.mixture
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.neural_network
import sklearn.preprocessing
import sklearn.random_projection
import sklearn.semi_supervised
import sklearn.svm
import sklearn.tree

# syft relative
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from ..util import generic_update_ast

LIB_NAME = "sklearn"
PACKAGE_SUPPORT = {"lib": LIB_NAME}

SKLEARN_MODULES = [
    sklearn,
    sklearn.ensemble,
    sklearn.linear_model,
    sklearn.kernel_ridge,
    sklearn.dummy,
    sklearn.discriminant_analysis,
    sklearn.decomposition,
    sklearn.cross_decomposition,
    sklearn.covariance,
    sklearn.compose,
    sklearn.cluster,
    sklearn.base,
    sklearn.tree,
    sklearn.dummy,
    sklearn.svm,
    sklearn.semi_supervised,
    sklearn.random_projection,
    sklearn.preprocessing,
    sklearn.neural_network,
    sklearn.neighbors,
    sklearn.naive_bayes,
    sklearn.mixture]

TARGET_METHODS = [
    'fit',
    'predict',
    'predict_log_proba',
    'predict_proba',
    'fit_transform',
    'transform',
]


def create_ast(client: TypeAny = None) -> Globals:
    ast = Globals(client)

    modules: TypeList[TypeTuple[str, TypeAny]] =\
        [(module.__name__, module) for module in SKLEARN_MODULES]

    sklearn_classes = [[(module.__name__ + "." + attribute,
                         module.__name__ + "." + attribute,
                         getattr(module, attribute))
                        for attribute in dir(module)
                        if str.isupper(attribute[0]) and "VALID_METRICS" not in attribute]
                       for module in SKLEARN_MODULES]
    sklearn_classes = sum(sklearn_classes, [])
    classes: TypeList[TypeTuple[str, str, TypeAny]] = sklearn_classes

    sklearn_methods = []
    for _, clas_name, clas in sklearn_classes:
        for method in dir(clas):
            if method in TARGET_METHODS:
                if method == "fit":
                    sklearn_methods.append((clas_name + "." + method,
                                            clas_name))
                else:
                    sklearn_methods.append((clas_name + "." + method,
                                            "pandas.DataFrame"))
    methods: TypeList[TypeTuple[str, str]] = sklearn_methods

    add_modules(ast, modules)
    add_classes(ast, classes)
    add_methods(ast, methods)

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_storable_object_attr_convenience_methods()

    return ast


update_ast = functools.partial(generic_update_ast, LIB_NAME, create_ast)
