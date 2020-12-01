# stdlib
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# syft relative
from . import attribute  # noqa: F401
from . import callable  # noqa: F401
from . import function  # noqa: F401
from . import globals  # noqa: F401
from . import klass  # noqa: F401
from . import method  # noqa: F401
from . import module  # noqa: F401


def get_parent(path: str, root: TypeAny) -> module.Module:
    parent = root
    for step in path.split(".")[:-1]:
        parent = parent.attrs[step]
    return parent


def add_modules(ast: globals.Globals, modules: TypeList[str]) -> None:
    for target_module in modules:
        parent = get_parent(target_module, ast)
        attr_name = target_module.rsplit(".", 1)[-1]
        parent.add_attr(
            attr_name=attr_name,
            attr=module.Module(
                name=attr_name,
                path_and_name=target_module,
                ref=None,
                return_type_name="",
            ),
        )


def add_classes(
    ast: globals.Globals, paths: TypeList[TypeTuple[str, str, TypeAny]]
) -> None:
    for path, return_type, ref in paths:
        parent = get_parent(path, ast)
        attr_name = path.rsplit(".", 1)[-1]
        parent.add_attr(
            attr_name=attr_name,
            attr=klass.Class(
                name=attr_name,
                path_and_name=path,
                ref=ref,
                return_type_name=return_type,
            ),
        )


def add_methods(ast: globals.Globals, paths: TypeList[TypeTuple[str, str]]) -> None:
    for path, return_type in paths:
        parent = get_parent(path, ast)
        path_list = path.split(".")
        parent.add_path(
            path=path_list, index=len(path_list) - 1, return_type_name=return_type
        )
