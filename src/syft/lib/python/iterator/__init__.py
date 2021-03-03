# stdlib
from typing import Any as TypeAny

# syft relative
from ....ast import add_classes
from ....ast import add_methods
from ....ast import add_modules
from ....ast.globals import Globals
from .templatable_iterator import TemplateableIterator
from .templated_iterator import Iterator
from .templated_iterator import type_cache


def create_iterator_ast(lib_ast: Globals, client: TypeAny = None) -> Globals:
    ast = Globals(client)

    modules = ["syft", "syft.lib", "syft.lib.python", "syft.lib.python.iterator"]
    classes = []
    methods = []

    for klass_name, (underlying_type_path, skip_typecheck) in type_cache.items():
        underlying_type = lib_ast.query(underlying_type_path).object_ref
        klass, allowlist = Iterator(underlying_type, skip_typecheck)
        globals()[klass_name.rsplit(".", 1)[-1]] = klass
        classes.append(
            (
                klass.__qualname__,
                klass.__qualname__,
                klass,
            )
        )

        for target_method, return_type in allowlist.items():
            methods.append(
                (
                    target_method,
                    return_type,
                )
            )

    add_modules(ast, modules)
    add_classes(ast, classes)
    add_methods(ast, methods)

    for ast_klass in ast.classes:
        ast_klass.create_pointer_class()
        ast_klass.create_send_method()
        ast_klass.create_storable_object_attr_convenience_methods()

    return ast
