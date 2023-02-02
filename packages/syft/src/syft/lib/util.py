""" A set of util methods used by the syft.lib submodule. """
# stdlib
import re
from typing import Callable
from typing import Union as TypeUnion

# relative
from ..ast.globals import Globals
from ..core.node.abstract.node import AbstractNodeClient

# this gets called on global ast as well as clients
# anything which wants to have its ast updated and has an add_attr method


def generic_update_ast(
    lib_name: str,
    create_ast: Callable,
    ast_or_client: TypeUnion[Globals, AbstractNodeClient],
) -> None:
    """
    Update the abstract syntax tree data structure used by the Globals or AbstractNodeClient data_subjects.

    Args:
        lib_name (str): Library name to update ast.
        create_ast (Callable): AST generation function for the given library.
        ast_or_client (Union[Globals, AbstractNodeClient]): DataSubject to update ast.

    Raises:
        ValueError: raises a ValueError exception if ast_or_client isn't a Globals or AbstractNodeClient instance.
    """
    if isinstance(ast_or_client, Globals):
        ast = ast_or_client
        new_lib_ast = create_ast(None)
        ast.add_attr(attr_name=lib_name, attr=new_lib_ast.attrs[lib_name])
    elif isinstance(ast_or_client, AbstractNodeClient):
        client = ast_or_client
        new_lib_ast = create_ast(client)
        client.lib_ast.attrs[lib_name] = new_lib_ast.attrs[lib_name]
        setattr(client, lib_name, new_lib_ast.attrs[lib_name])
    else:
        raise ValueError(
            f"Expected param of type (Globals, AbstractNodeClient), but got {type(ast_or_client)}"
        )


def full_name_with_qualname(klass: type) -> str:
    """Returns the klass module name + klass qualname."""
    try:
        if not hasattr(klass, "__module__"):
            return f"builtins.{get_qualname_for(klass)}"
        return f"{klass.__module__}.{get_qualname_for(klass)}"
    except Exception:
        # try name as backup
        print("Failed to get FQN for:", klass, type(klass))
    return full_name_with_name(klass=klass)


def full_name_with_name(klass: type) -> str:
    """Returns the klass module name + klass name."""
    try:
        if not hasattr(klass, "__module__"):
            return f"builtins.{get_name_for(klass)}"
        return f"{klass.__module__}.{get_name_for(klass)}"
    except Exception as e:
        print("Failed to get FQN for:", klass, type(klass))
        raise e


def get_qualname_for(klass: type):
    qualname = getattr(klass, "__qualname__", None) or getattr(klass, "__name__", None)
    if qualname is None:
        qualname = extract_name(klass)
    return qualname


def get_name_for(klass: type):
    klass_name = getattr(klass, "__name__", None)
    if klass_name is None:
        klass_name = extract_name(klass)
    return klass_name


def extract_name(klass: type):
    name_regex = r".+class.+?([\w\._]+).+"
    regex2 = r"([\w\.]+)"
    matches = re.match(name_regex, str(klass))
    if matches is None:
        matches = re.match(regex2, str(klass))
    try:
        fqn = matches[1]
        if "." in fqn:
            return fqn.split(".")[-1]
        return fqn
    except Exception as e:
        print(f"Failed to get klass name {klass}")
        raise e
