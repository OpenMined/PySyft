# stdlib
from typing import Any as TypeAny
from typing import Dict
from typing import Union
from typing import List as TypeList
from typing import Tuple as TypeTuple
import functools

# syft relative
# from . import batchencoding # noqa: 401
from . import tokenizer # noqa: 401
# from .allowlist import allowlist
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

def get_return_type(support_dict: Union[str, Dict[str, str]]) -> str:
    if isinstance(support_dict, str):
        return support_dict
    else:
        return support_dict["return_type"]


def create_ast(client: TypeAny = None) -> Globals:

    ast = Globals(client=client)

    import transformers

    # Define which transformer modules to add to the AST
    modules: TypeList[TypeTuple[str, TypeAny]] = [
        ("transformers", transformers),
        ("transformers.tokenization_utils_base", transformers.tokenization_utils_base)
    ]

    # Define which transformer classes to add to the AST
    classes: TypeList[TypeTuple[str, str, TypeAny]] = [
        ("transformers.tokenization_utils_base.BatchEncoding",
         "transformers.tokenization_utils_base.BatchEncoding",
         transformers.tokenization_utils_base.BatchEncoding),
        ("transformers.PreTrainedTokenizerFast",
         "transformers.PreTrainedTokenizerFast",
         transformers.PreTrainedTokenizerFast),
    ]


    # Define which methods to add to the AST
    methods: TypeList[TypeTuple[str, str]] = [
        ("transformers.PreTrainedTokenizerFast.__call__",
         "transformers.tokenization_utils_base.BatchEncoding")
    ]

    add_modules(ast, modules)
    add_classes(ast, classes)
    add_methods(ast, methods)

    # for method, return_type_name_or_dict in allowlist.items():
    #     # TODO Add version_supported checks.

    #     return_type = get_return_type(support_dict=return_type_name_or_dict)
    #     if return_type == "unknown":
    #         # this allows us to import them for testing
    #         continue
    #     ast.add_path(
    #         path=method, framework_reference=transformers, return_type_name=return_type
    #     )


    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_storable_object_attr_convenience_methods()

    return ast


update_ast = functools.partial(generic_update_ast, LIB_NAME, create_ast)
