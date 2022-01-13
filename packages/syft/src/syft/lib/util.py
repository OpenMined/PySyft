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


def get_original_constructor_name(object_name: str) -> str:
    """
    Generate name for original constructor

    For each custom constructor, we move the original constructor to a consistent location relative to
    the original constructor so that each custom constructor automatically knows where to find the original
    method it is overloading. Namely, we move the original constructor to a different attr within the same
    module as the original constructor. This method specifies the naming convention that we use to name
    the original constructor when it is moved.

    :param str object_name: the original constructor's original name
    :return: a modified string containing the original_{object_name}
    :rtype: str
    """
    return f"original_{object_name}"


def replace_classes_in_module(
    module: ModuleType,
    from_class: Callable,
    to_class: Callable,
    ignore_prefix: Optional[str] = None,
) -> None:
    """
    Recursively replace occurrence of `from_class` to `to_class` inside module.

    For example, when syft replaces torch.nn.parameter.Parameter constructor,
    there's also need to replace the same constructor in other modules that has already
    imported it.

    :params ModuleType module: top-level module to traverse.
    :params Callable from_class: Original constructor.
    :param Callable to_class: syft's ObjectConstructor.
    :param str ignore_prefix: string value containing a prefix that should be ignored.
    """
    visited_modules = []

    # inner function definition to update a module recursively
    def recursive_update(
        module: ModuleType, attr_name: Union[str, None] = None
    ) -> None:
        """
        Updates a specific module recursively.

        :param ModuleType module: module to be updated.
        :param str attr_name: module's attribute name to be updated. 
        """
        # check if we need to skip this attribute to preserve our unmodified
        # original copy
        if (
            attr_name is not None
            and ignore_prefix is not None
            and attr_name.startswith(ignore_prefix)
        ):
            # found an attr that should be skipped so lets return
            return
        attr = getattr(module, attr_name) if isinstance(attr_name, str) else module
        if isinstance(attr, ModuleType) and attr not in visited_modules:
            visited_modules.append(attr)
            for child_attr_name in dir(attr):
                recursive_update(attr, child_attr_name)
        elif (
            isinstance(attr_name, str) and inspect.isclass(attr) and attr is from_class
        ):
            setattr(module, attr_name, to_class)

    recursive_update(module)


def full_name_with_qualname(klass: type) -> str:
    """ Returns the klass module name + klass qualname."""
    return f"{klass.__module__}.{klass.__qualname__}"


def full_name_with_name(klass: type) -> str:
    """ Returns the klass module name + klass name.  """
    return f"{klass.__module__}.{klass.__name__}"
