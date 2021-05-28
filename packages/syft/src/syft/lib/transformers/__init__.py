# stdlib
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple
import functools

# syft relative
from . import batchencoding # noqa: 401
# from . import tokenizer # noqa: 401
from ..util import generic_update_ast
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals

# The library name
LIB_NAME = "transformers"
PACKAGE_SUPPORT = {
    "lib": LIB_NAME,
}


def create_ast(client: TypeAny = None) -> Globals:

    import transformers

    ast = Globals(client=client)

    # Define which transformer modules to add to the AST
    modules: TypeList[TypeTuple[str, TypeAny]] = [
        ("transformers", transformers)
    ]

    # Define which transformer classes to add to the AST
    classes: TypeList[TypeTuple[str, str, TypeAny]] = [
        ("transformers.tokenization_utils_base.BatchEncoding",
         "transformers.tokenization_utils_base.BatchEncoding",
         transformers.tokenization_utils_base.BatchEncoding),
        ("transformers.PreTrainedTokenizerFast",
         "transformers.PreTrainedTokenizerFast",
         transformers.PreTrainedTokenizerFast)
    ]

    # Define which methods to add to the AST
    methods: TypeList[TypeTuple[str, str]] = [
        ("transformers.PreTrainedTokenizerFast.__call__",
         "transformers.tokenization_utils_base.BatchEncoding")
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
