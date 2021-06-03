# stdlib
import functools
from typing import Any as TypeAny
from typing import Dict
from typing import List as TypeList
from typing import Tuple as TypeTuple
from typing import Union

# syft relative
from . import batchencoding  # noqa: 401
from . import tokenizer  # noqa: 401
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals

# from .allowlist import allowlist
from ..util import generic_update_ast

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

    # third party
    import transformers

    # Define which transformer modules to add to the AST
    modules: TypeList[TypeTuple[str, TypeAny]] = [
        ("transformers", transformers),
        ("transformers.tokenization_utils_base", transformers.tokenization_utils_base),
    ]

    # Define which transformer classes to add to the AST
    classes: TypeList[TypeTuple[str, str, TypeAny]] = [
        (
            "transformers.tokenization_utils_base.BatchEncoding",
            "transformers.tokenization_utils_base.BatchEncoding",
            transformers.tokenization_utils_base.BatchEncoding,
        ),
        (
            "transformers.PreTrainedTokenizerFast",
            "transformers.PreTrainedTokenizerFast",
            transformers.PreTrainedTokenizerFast,
        ),
    ]

    # Define which methods to add to the AST
    methods: TypeList[TypeTuple[str, str]] = [
        (
            "transformers.PreTrainedTokenizerFast.__call__",
            "transformers.tokenization_utils_base.BatchEncoding",
        ),
        (
            "transformers.tokenization_utils_base.BatchEncoding.__getitem__",
            "torch.Tensor",
        ),
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
