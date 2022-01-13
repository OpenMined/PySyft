''' A set of util methods used by the syft.lib submodule. '''
# stdlib
import inspect
from types import ModuleType
from typing import Callable
from typing import Optional
from typing import Union
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
    Update the abstract syntax tree data structure used by the Globals or AbstractNodeClient entities.

    :param str lib_name: 
    :param Callable create_ast:
    :param Union[Globals, AbstractNodeClient] ast_or_client:

    :raises ValueError: raises a ValueError exception if ast_or_client isn't a Globals or AbstractNodeClient instance.
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


def is_static_method(klass: type, attr: str) -> bool:
    """
    Test if a value of a class is static method.

    Example:

    .. code-block::

        class MyClass(object):
            @staticmethod
            def method():
                ...

    
    :param type klass: the class on which we want to check whether the method is statically implemented
    :param str attr: the name of the method we want to check.

    :return: whether or not a method named <attr> is a static method of class <klass>
    :rtype: bool
    """

    if not inspect.isclass(klass):
        return False

    if hasattr(klass, attr):
        value = getattr(klass, attr)
    else:
        return False

    if getattr(klass, attr) != value:
        raise AttributeError("Method don't match")

    for cls in inspect.getmro(klass):
        if inspect.isroutine(value):
            if attr in cls.__dict__:
                bound_value = cls.__dict__[attr]
                if isinstance(bound_value, staticmethod):
                    return True
    return False


def copy_static_methods(from_class: type, to_class: type) -> None:
    """
    Copies all static methods from one class to another class

    This utility was initialized during the creation of the Constructor for PyTorch's "th.Tensor" class. Since we
    replace each original constructor (th.Tensor) with on we implement (torch.UppercaseTensorConstructor), we also
    need to make sure that our new constructor has any static methods which were previously stored on th.Tensor.
    Otherwise, the library might look for them there, not find them, and then trigger an error.

    :param type from_class: the class on which we look for static methods co copy
    :param type to_class: the class onto which we copy all static methods found in <from_class>

    """
    # there are no static methods if from_class itself is not a type (sometimes funcs get passed in)

    for attr in dir(from_class):
        if is_static_method(klass=from_class, attr=attr):
            setattr(to_class, attr, getattr(from_class, attr))




def full_name_with_qualname(klass: type) -> str:
    """ Returns the klass module name + klass qualname."""
    return f"{klass.__module__}.{klass.__qualname__}"


def full_name_with_name(klass: type) -> str:
    """ Returns the klass module name + klass name.  """
    return f"{klass.__module__}.{klass.__name__}"
