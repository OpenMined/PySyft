# stdlib
from typing import Dict
from typing import Union

# syft relative
from ... import ast as astlib
from ...ast.globals import Globals
from .int import Int


def get_return_type(support_dict: Union[str, Dict[str, str]]) -> str:
    if isinstance(support_dict, str):
        return support_dict
    else:
        return support_dict["return_type"]


def version_supported(support_dict: Union[str, Dict[str, str]]) -> bool:
    return True


def create_python_ast() -> Globals:
    ast = Globals()

    # build fake syft.lib.python module
    syft = lambda: None  # noqa: E731
    lib = lambda: None  # noqa: E731
    python = lambda: None  # noqa: E731
    setattr(lib, "python", python)
    setattr(python, "Int", Int)
    setattr(syft, "lib", lib)

    ast.attrs["syft"] = astlib.module.Module(
        name="syft",
        path_and_name="syft",
        ref=syft,
        return_type_name="",
    )

    syft_node = ast.attrs["syft"]
    add_attr = getattr(syft_node, "add_attr", None)
    if add_attr is not None:
        add_attr(
            attr_name="lib",
            attr=astlib.module.Module(
                "lib",
                "syft.lib",
                syft,
                return_type_name="",
            ),
        )

    syft_node_attrs = getattr(syft_node, "attrs", None)
    if syft_node_attrs is not None:
        lib_node = syft_node_attrs["lib"]
        add_attr = getattr(lib_node, "add_attr", None)
        if add_attr is not None:
            add_attr(
                attr_name="python",
                attr=astlib.module.Module(
                    "python",
                    "syft.lib.python",
                    lib,
                    return_type_name="",
                ),
            )

    python_node = lib_node.attrs["python"]
    add_attr = getattr(python_node, "add_attr", None)
    if add_attr is not None:
        add_attr(
            attr_name="Int",
            attr=astlib.klass.Class(
                "Int",
                "syft.lib.python.Int",
                Int,
                return_type_name="syft.lib.python.Int",
            ),
        )

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_serialization_methods()
        klass.create_storable_object_attr_convenience_methods()

    return ast
