# stdlib
from typing import Any as TypeAny
from typing import Callable
from typing import Dict
from typing import List as TypeList
from typing import Tuple as TypeTuple
from typing import Union

# third party
import sympc

# syft relative
from ...ast.globals import Globals
from ...ast.klass import Class
from ...ast.module import Module
from ..python.primitive_container import Any
from .session import SySession


def get_return_type(support_dict: Union[str, Dict[str, str]]) -> str:
    if isinstance(support_dict, str):
        return support_dict
    else:
        return support_dict["return_type"]


def get_parent(path: str, root: TypeAny) -> Module:
    parent = root
    for step in path.split(".")[:-1]:
        parent = parent.attrs[step]
    return parent


def add_modules(ast: Globals, modules: TypeList[TypeTuple[str, Callable]]) -> None:
    for module, ref in modules:
        parent = get_parent(module, ast)
        attr_name = module.rsplit(".", 1)[-1]

        parent.add_attr(
            attr_name=attr_name,
            attr=Module(
                attr_name,
                module,
                ref=ref,
                return_type_name="",
            ),
        )


def add_classes(ast: Globals, paths: TypeList[TypeTuple[str, str, Any]]) -> None:
    for path, return_type, ref in paths:
        parent = get_parent(path, ast)
        attr_name = path.rsplit(".", 1)[-1]

        parent.add_attr(
            attr_name=attr_name,
            attr=Class(
                attr_name,
                path,
                ref,  # type: ignore
                return_type_name=return_type,
            ),
        )


def add_functions(ast: Globals, paths: TypeList[TypeTuple[str, str]]) -> None:
    for path, return_type in paths:
        parent = get_parent(path, ast)
        path_list = path.split(".")

        parent.add_path(
            path=path_list, index=len(path_list) - 1, return_type_name=return_type
        )


def create_sympc_ast() -> Globals:
    ast = Globals()

    modules = [
        ("sympc", sympc),
        ("sympc.session", sympc.session),
        ("sympc.tensor", sympc.tensor),
        ("sympc.protocol", sympc.protocol.spdz),
        ("sympc.protocol.spdz", sympc.protocol.spdz),
    ]

    classes = [
        ("sympc.session.Session", "sympc.session.Session", SySession),
    ]

    functions = [("sympc.protocol.spdz.mul_parties", "torch.Tensor")]

    add_modules(ast, modules)
    add_classes(ast, classes)
    add_functions(ast, functions)

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_serialization_methods()
        klass.create_storable_object_attr_convenience_methods()

    return ast
