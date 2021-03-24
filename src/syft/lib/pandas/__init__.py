# stdlib
import functools
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# third party
from packaging import version
import pandas as pd

# syft relative
from . import frame  # noqa: 401
from . import series  # noqa: 401
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from ..misc.union import UnionGenerator
from ..util import generic_update_ast

LIB_NAME = "pandas"
PACKAGE_SUPPORT = {"lib": LIB_NAME}

LIB_VERSION = version.parse(pd.__version__.split("+")[0])


def create_ast(client: TypeAny = None) -> Globals:
    ast = Globals(client)

    modules: TypeList[TypeTuple[str, TypeAny]] = [("pandas", pd)]

    classes: TypeList[TypeTuple[str, str, TypeAny]] = [
        ("pandas.DataFrame", "pandas.DataFrame", pd.DataFrame),
        ("pandas.Series", "pandas.Series", pd.Series),
    ]

    methods: TypeList[TypeTuple[str, str]] = [
        ("pandas.read_csv", "pandas.DataFrame"),
        ("pandas.DataFrame.__getitem__", "pandas.Series"),
        ("pandas.DataFrame.__setitem__", "pandas.Series"),
        ("pandas.DataFrame.__len__", "syft.lib.python.Int"),
        ("pandas.DataFrame.__abs__", "pandas.DataFrame"),
        ("pandas.DataFrame.__add__", "pandas.DataFrame"),
        ("pandas.DataFrame.__and__", "pandas.DataFrame"),
        ("pandas.DataFrame.__eq__", "pandas.DataFrame"),
        ("pandas.DataFrame.__floordiv__", "pandas.DataFrame"),
        ("pandas.DataFrame.__ge__", "pandas.DataFrame"),
        ("pandas.DataFrame.__gt__", "pandas.DataFrame"),
        ("pandas.DataFrame.__iadd__", "pandas.DataFrame"),
        ("pandas.DataFrame.__iand__", "pandas.DataFrame"),
        ("pandas.DataFrame.__ifloordiv__", "pandas.DataFrame"),
        ("pandas.DataFrame.__imod__", "pandas.DataFrame"),
        ("pandas.DataFrame.__imul__", "pandas.DataFrame"),
        ("pandas.DataFrame.__ipow__", "pandas.DataFrame"),
        ("pandas.DataFrame.__isub__", "pandas.DataFrame"),
        ("pandas.DataFrame.__le__", "pandas.DataFrame"),
        ("pandas.DataFrame.__lt__", "pandas.DataFrame"),
        ("pandas.DataFrame.__mod__", "pandas.DataFrame"),
        ("pandas.DataFrame.__mul__", "pandas.DataFrame"),
        ("pandas.DataFrame.__ne__", "pandas.DataFrame"),
        ("pandas.DataFrame.__neg__", "pandas.DataFrame"),
        ("pandas.DataFrame.__pos__", "pandas.DataFrame"),
        ("pandas.DataFrame.__pow__", "pandas.DataFrame"),
        ("pandas.DataFrame.__rfloordiv__", "pandas.DataFrame"),
        ("pandas.DataFrame.__rmod__", "pandas.DataFrame"),
        ("pandas.DataFrame.__rmul__", "pandas.DataFrame"),
        ("pandas.DataFrame.__round__", "pandas.DataFrame"),
        ("pandas.DataFrame.__rpow__", "pandas.DataFrame"),
        ("pandas.DataFrame.__rsub__", "pandas.DataFrame"),
        ("pandas.DataFrame.__rtruediv__", "pandas.DataFrame"),
        ("pandas.DataFrame.__sub__", "pandas.DataFrame"),
        ("pandas.DataFrame.__truediv__", "pandas.DataFrame"),
        ("pandas.DataFrame.dropna", "pandas.DataFrame"),
        ("pandas.Series.__getitem__", "pandas.Series"),
        ("pandas.Series.__setitem__", "pandas.Series"),
        ("pandas.Series.__len__", "syft.lib.python.Int"),
        ("pandas.Series.__abs__", "pandas.Series"),
        ("pandas.Series.__add__", "pandas.Series"),
        ("pandas.Series.__and__", "pandas.Series"),
        ("pandas.Series.__divmod__", "pandas.Series"),
        ("pandas.Series.__eq__", "pandas.Series"),
        ("pandas.Series.__floordiv__", "pandas.Series"),
        ("pandas.Series.__ge__", "pandas.Series"),
        ("pandas.Series.__gt__", "pandas.Series"),
        ("pandas.Series.__iadd__", "pandas.Series"),
        ("pandas.Series.__iand__", "pandas.Series"),
        ("pandas.Series.__ifloordiv__", "pandas.Series"),
        ("pandas.Series.__imod__", "pandas.Series"),
        ("pandas.Series.__imul__", "pandas.Series"),
        ("pandas.Series.__ipow__", "pandas.Series"),
        ("pandas.Series.__isub__", "pandas.Series"),
        ("pandas.Series.__le__", "pandas.Series"),
        ("pandas.Series.__lt__", "pandas.Series"),
        ("pandas.Series.__mod__", "pandas.Series"),
        ("pandas.Series.__mul__", "pandas.Series"),
        ("pandas.Series.__ne__", "pandas.Series"),
        ("pandas.Series.__neg__", "pandas.Series"),
        ("pandas.Series.__pos__", "pandas.Series"),
        ("pandas.Series.__pow__", "pandas.Series"),
        ("pandas.Series.__rdivmod__", "pandas.Series"),
        ("pandas.Series.__rfloordiv__", "pandas.Series"),
        ("pandas.Series.__rmod__", "pandas.Series"),
        ("pandas.Series.__rmul__", "pandas.Series"),
        ("pandas.Series.__round__", "pandas.Series"),
        ("pandas.Series.__rpow__", "pandas.Series"),
        ("pandas.Series.__rsub__", "pandas.Series"),
        ("pandas.Series.__rtruediv__", "pandas.Series"),
        ("pandas.Series.__sub__", "pandas.Series"),
        ("pandas.Series.__truediv__", "pandas.Series"),
        ("pandas.Series.add", "pandas.Series"),
        ("pandas.Series.sub", "pandas.Series"),
        ("pandas.Series.mul", "pandas.Series"),
        ("pandas.Series.div", "pandas.Series"),
        ("pandas.Series.truediv", "pandas.Series"),
        ("pandas.Series.floordiv", "pandas.Series"),
        ("pandas.Series.mod", "pandas.Series"),
        ("pandas.Series.pow", "pandas.Series"),
        ("pandas.Series.radd", "pandas.Series"),
        ("pandas.Series.rsub", "pandas.Series"),
        ("pandas.Series.rmul", "pandas.Series"),
        ("pandas.Series.rdiv", "pandas.Series"),
        ("pandas.Series.rtruediv", "pandas.Series"),
        ("pandas.Series.rfloordiv", "pandas.Series"),
        ("pandas.Series.rmod", "pandas.Series"),
        ("pandas.Series.rpow", "pandas.Series"),
        ("pandas.Series.lt", "pandas.Series"),
        ("pandas.Series.gt", "pandas.Series"),
        ("pandas.Series.le", "pandas.Series"),
        ("pandas.Series.ge", "pandas.Series"),
        ("pandas.Series.ne", "pandas.Series"),
        ("pandas.Series.eq", "pandas.Series"),
        ("pandas.Series.argsort", "pandas.Series"),
        ("pandas.Series.round", "pandas.Series"),
        ("pandas.Series.head", "pandas.Series"),
        ("pandas.Series.tail", "pandas.Series"),
        ("pandas.Series.any", "syft.lib.python.Bool"),
        ("pandas.Series.shape", "syft.lib.python.Tuple"),
        ("pandas.Series.all", "syft.lib.python.Bool"),
        ("pandas.Series.argmax", "syft.lib.python.Int"),
        ("pandas.Series.nbytes", "syft.lib.python.Int"),
        ("pandas.Series.mean", "syft.lib.python.Float"),
        ("pandas.Series.ndim", "syft.lib.python.Int"),
        ("pandas.Series.size", "syft.lib.python.Int"),
        ("pandas.Series.hasnans", "syft.lib.python.Bool"),
        ("pandas.Series.empty", "syft.lib.python.Bool"),
        ("pandas.Series.T", "pandas.Series"),
        ("pandas.Series.dropna", "pandas.Series"),
        ("pandas.Series.to_frame", "pandas.DataFrame"),
        ("pandas.Series.to_list", "syft.lib.python.List"),
        (
            "pandas.Series.sum",
            UnionGenerator["syft.lib.python.Float", "syft.lib.python.Int"],
        ),
        ("pandas.Series.median", "syft.lib.python.Float"),
        (
            "pandas.Series.max",
            UnionGenerator[
                "syft.lib.python.Bool", "syft.lib.python.Float", "syft.lib.python.Int"
            ],
        ),
        (
            "pandas.Series.min",
            UnionGenerator[
                "syft.lib.python.Bool", "syft.lib.python.Float", "syft.lib.python.Int"
            ],
        ),
    ]

    if LIB_VERSION > version.parse("1.2.0"):
        methods += [
            ("pandas.DataFrame.__divmod__", "pandas.DataFrame"),
            ("pandas.DataFrame.__rdivmod__", "pandas.DataFrame"),
        ]

    add_modules(ast, modules)
    add_classes(ast, classes)
    add_methods(ast, methods)

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        # TODO: Pandas can't have tags and description because they break the dict
        # klass.create_storable_object_attr_convenience_methods()

    return ast


# we cant create Unions that refer to the package itself until the create_ast
# has completed first so we can call again into post_update_ast to finish these
# TODO: add support for self referential unions using some kind of post update
# Issue: https://github.com/OpenMined/PySyft/issues/5323
def post_create_ast(ast: Globals) -> Globals:
    self_referencing_methods = [
        ("pandas.Series.loc", UnionGenerator["pandas.DataFrame", "pandas.Series"]),
        ("pandas.Series.iloc", UnionGenerator["pandas.DataFrame", "pandas.Series"]),
        ("pandas.DataFrame.loc", UnionGenerator["pandas.DataFrame", "pandas.Series"]),
        ("pandas.DataFrame.iloc", UnionGenerator["pandas.DataFrame", "pandas.Series"]),
    ]

    add_methods(ast, self_referencing_methods)

    return ast


update_ast = functools.partial(generic_update_ast, LIB_NAME, create_ast)
# post_update_ast = functools.partial(generic_update_ast, LIB_NAME, post_create_ast)
