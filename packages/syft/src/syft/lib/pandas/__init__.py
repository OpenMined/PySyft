# stdlib
import functools
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# third party
from packaging import version
import pandas as pd

# syft relative
from . import categorical  # noqa: 401
from . import categorical_dtype  # noqa: 401
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
        ("pandas.CategoricalDtype", "pandas.CategoricalDtype", pd.CategoricalDtype),
        ("pandas.Categorical", "pandas.Categorical", pd.Categorical),
    ]

    methods: TypeList[TypeTuple[str, str]] = [
        ("pandas.json_normalize", "pandas.DataFrame"),
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
        ("pandas.DataFrame.apply", "pandas.DataFrame"),
        ("pandas.DataFrame.loc", UnionGenerator["pandas.DataFrame", "pandas.Series"]),
        ("pandas.DataFrame.iloc", UnionGenerator["pandas.DataFrame", "pandas.Series"]),
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
        ("pandas.Series.apply", "pandas.Series"),
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
        ("pandas.Series.loc", UnionGenerator["pandas.DataFrame", "pandas.Series"]),
        ("pandas.Series.iloc", UnionGenerator["pandas.DataFrame", "pandas.Series"]),
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
        # ===== methods for pd.Categorical ======
        # ("pandas.Categorical.__array__",),
        # ("pandas.Categorical.__array_ufunc__",),
        # ("pandas.Categorical.__contains__",),
        # ("pandas.Categorical.__dir__",),
        # ("pandas.Categorical.__init__",),
        # ("pandas.Categorical.__iter__",), Return a list Iterator
        # ("pandas.Categorical.__repr__",),
        ("pandas.Categorical.__eq__", "numpy.ndarray"),
        ("pandas.Categorical.__ge__", "numpy.ndarray"),
        ("pandas.Categorical.__gt__", "numpy.ndarray"),
        ("pandas.Categorical.__le__", "numpy.ndarray"),
        ("pandas.Categorical.__lt__", "numpy.ndarray"),
        ("pandas.Categorical.__ne__", "numpy.ndarray"),
        ("pandas.Categorical.__len__", "syft.lib.python.Int"),
        (
            "pandas.Categorical.__getitem__",
            UnionGenerator[
                "syft.lib.python.Bool",
                "syft.lib.python.Float",
                "syft.lib.python.Int",
                "syft.lib.python.Complex",
            ],
        ),
        ("pandas.Categorical.__setitem__", "syft.lib.python._SyNone"),
        # ("pandas.Categorical.__setstate__",), # method to support Pickle support
        # ("pandas.Categorical.__sizeof__",),
        ("pandas.Categorical.add_categories", "pandas.Categorical"),
        ("pandas.Categorical.argmax", "syft.lib.python.Int"),
        ("pandas.Categorical.argmin", "syft.lib.python.Int"),
        ("pandas.Categorical.argsort", "numpy.ndarray"),
        ("pandas.Categorical.as_ordered", "pandas.Categorical"),
        ("pandas.Categorical.as_unordered", "pandas.Categorical"),
        # ("pandas.Categorical.astype",), # Need support for other pandas types
        # ("pandas.Categorical.check_for_ordered",),
        ("pandas.Categorical.copy", "pandas.Categorical"),
        ("pandas.Categorical.describe", "pandas.DataFrame"),
        ("pandas.Categorical.dropna", "pandas.Categorical"),
        ("pandas.Categorical.equals", "syft.lib.python.Bool"),
        (
            "pandas.Categorical.factorize",
            "syft.lib.python.Tuple",
        ),
        ("pandas.Categorical.fillna", "pandas.Categorical"),
        ("pandas.Categorical.from_codes", "pandas.Categorical"),
        ("pandas.Categorical.is_dtype_equal", "syft.lib.python.Bool"),
        ("pandas.Categorical.isin", "numpy.ndarray"),
        ("pandas.Categorical.isna", "numpy.ndarray"),
        ("pandas.Categorical.isnull", "numpy.ndarray"),
        (
            "pandas.Categorical.map",
            "pandas.Categorical",
        ),  # TODO: returns pd.Categorical or pd.Index
        (
            "pandas.Categorical.max",
            UnionGenerator[
                "syft.lib.python.Bool",
                "syft.lib.python.Float",
                "syft.lib.python.Int",
                "syft.lib.python.Complex",
            ],
        ),
        ("pandas.Categorical.memory_usage", "syft.lib.python.Int"),
        (
            "pandas.Categorical.min",
            UnionGenerator[
                "syft.lib.python.Bool",
                "syft.lib.python.Float",
                "syft.lib.python.Int",
                "syft.lib.python.Complex",
            ],
        ),
        ("pandas.Categorical.mode", "pandas.Categorical"),
        ("pandas.Categorical.notna", "numpy.ndarray"),
        ("pandas.Categorical.notnull", "numpy.ndarray"),
        ("pandas.Categorical.ravel", "pandas.Categorical"),
        ("pandas.Categorical.remove_categories", "pandas.Categorical"),
        ("pandas.Categorical.remove_unused_categories", "pandas.Categorical"),
        (
            "pandas.Categorical.rename_categories",
            "pandas.Categorical",
        ),
        (
            "pandas.Categorical.reorder_categories",
            "pandas.Categorical",
        ),
        ("pandas.Categorical.repeat", "pandas.Categorical"),
        ("pandas.Categorical.replace", "pandas.Categorical"),
        ("pandas.Categorical.reshape", "pandas.Categorical"),
        ("pandas.Categorical.searchsorted", "syft.lib.python.Int"),
        (
            "pandas.Categorical.set_categories",
            "pandas.Categorical",
        ),
        (
            "pandas.Categorical.set_ordered",
            "pandas.Categorical",
        ),
        ("pandas.Categorical.shift", "pandas.Categorical"),
        ("pandas.Categorical.sort_values", "pandas.Categorical"),
        ("pandas.Categorical.take", "pandas.Categorical"),
        ("pandas.Categorical.take_nd", "pandas.Categorical"),
        ("pandas.Categorical.to_dense", "numpy.ndarray"),
        ("pandas.Categorical.to_list", "syft.lib.python.Float"),
        ("pandas.Categorical.to_numpy", "numpy.ndarray"),
        ("pandas.Categorical.tolist", "syft.lib.python.Float"),
        ("pandas.Categorical.unique", "pandas.Categorical"),
        ("pandas.Categorical.value_counts", "pandas.Series"),
        ("pandas.Categorical.view", "pandas.Categorical"),
        # ==== pd.Categorical properties ====
        ("pandas.Categorical.T", "pandas.Categorical"),
        # ("pandas.Categorical.categories",), require support for pd.Index
        ("pandas.Categorical.codes", "numpy.ndarray"),
        ("pandas.Categorical.dtype", "pandas.CategoricalDtype"),
        ("pandas.Categorical.nbytes", "syft.lib.python.Int"),
        ("pandas.Categorical.ordered", "syft.lib.python.Bool"),
        ("pandas.Categorical.shape", "syft.lib.python.Tuple"),
        ("pandas.Categorical.ndim", "syft.lib.python.Int"),
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
        # klass.create_storable_object_attr_convenience_methods()

    return ast


update_ast = functools.partial(generic_update_ast, LIB_NAME, create_ast)
