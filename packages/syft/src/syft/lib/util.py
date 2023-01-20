""" A set of util methods used by the syft.lib submodule. """
# stdlib
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
    return f"{klass.__module__}.{klass.__qualname__}"


def full_name_with_name(klass: type) -> str:
    """Returns the klass module name + klass name."""
    return f"{klass.__module__}.{klass.__name__}"
