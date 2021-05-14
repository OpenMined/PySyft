# stdlib
import functools
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# third party
import pytorch_lightning as pl

# syft relative
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from ..util import generic_update_ast

LIB_NAME = "pytorch_lightning"
PACKAGE_SUPPORT = {
    "lib": LIB_NAME,
    "torch": {"max_version": "1.8.1"},
    "python": {"max_version": (3, 9, 99)},
}


def create_ast(client: TypeAny) -> Globals:
    ast = Globals(client=client)

    modules: TypeList[TypeTuple[str, TypeAny]] = [
        ("pytorch_lightning", pl),
        ("pytorch_lightning.metrics", pl.metrics),
        ("pytorch_lightning.metrics.classification", pl.metrics.classification),
        (
            "pytorch_lightning.metrics.classification.accuracy",
            pl.metrics.classification.accuracy,
        ),
    ]

    classes: TypeList[TypeTuple[str, str, TypeAny]] = [
        ("pytorch_lightning.Trainer", "pytorch_lightning.Trainer", pl.Trainer),
        (
            "pytorch_lightning.metrics.classification.accuracy.Accuracy",
            "pytorch_lightning.metrics.classification.accuracy.Accuracy",
            pl.metrics.classification.accuracy.Accuracy,
        ),
    ]

    methods: TypeList[TypeTuple[str, str]] = [
        ("pytorch_lightning.Trainer.fit", "syft.lib.python._SyNone"),
        ("pytorch_lightning.Trainer.test", "syft.lib.python._SyNone"),
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
