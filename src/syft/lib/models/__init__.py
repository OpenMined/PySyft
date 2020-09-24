# stdlib
from typing import Any
from typing import Dict
from typing import List as TypeList
from typing import Tuple
from typing import Union

# syft relative
from ...ast.globals import Globals
from ...ast.klass import Class
from ...ast.module import Module
from .mnist import MNIST


def get_parent(path: str, root: Any) -> Module:
    parent = root
    for step in path.split(".")[:-1]:
        parent = parent.attrs[step]
    return parent


def add_modules(ast: Globals, modules: TypeList[str]) -> None:
    for module in modules:
        parent = get_parent(module, ast)
        attr_name = module.rsplit(".", 1)[-1]

        parent.add_attr(
            attr_name=attr_name,
            attr=Module(
                attr_name,
                module,
                None,
                return_type_name="",
            ),
        )


def add_classes(ast: Globals, paths: TypeList[Tuple[str, str, Any]]) -> None:
    for path, return_type, ref in paths:
        parent = get_parent(path, ast)
        attr_name = path.rsplit(".", 1)[-1]

        parent.add_attr(
            attr_name=attr_name,
            attr=Class(
                attr_name,
                path,
                ref,
                return_type_name=return_type,
            ),
        )


def add_methods(ast: Globals, paths: TypeList[Tuple[str, str, Any]]) -> None:
    for path, return_type, _ in paths:
        parent = get_parent(path, ast)

        parent.add_path(path=path.split("."), index=2, return_type_name=return_type)


def get_return_type(support_dict: Union[str, Dict[str, str]]) -> str:
    if isinstance(support_dict, str):
        return support_dict
    else:
        return support_dict["return_type"]


def create_models_ast() -> Globals:
    ast = Globals()

    modules = [
        "models",
    ]
    classes = [
        ("models.MNIST", "models.MNIST", MNIST),
    ]
    methods = [
        ("models.MNIST.__call__", "models.MNIST", MNIST),
        ("models.MNIST.parameters", "torch.nn.Parameter", MNIST),
        ("models.MNIST.to", "models.MNIST", MNIST),
    ]

    add_modules(ast, modules)
    add_classes(ast, classes)
    add_methods(ast, methods)

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_serialization_methods()
        klass.create_storable_object_attr_convenience_methods()

    return ast
