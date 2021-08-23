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
    """Test if a value of a class is static method.

    Example:

    .. code-block::

        class MyClass(object):
            @staticmethod
            def method():
                ...

    Args:
        klass (type): the class on which we want to check whether the method is statically implemented
        attr (str): the name of the method we want to check.

    Returns:
        bool: whether or not a method named <attr> is a static method of class <klass>
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
    """Copies all static methods from one class to another class

    This utility was initialized during the creation of the Constructor for PyTorch's "th.Tensor" class. Since we
    replace each original constructor (th.Tensor) with on we implement (torch.UppercaseTensorConstructor), we also
    need to make sure that our new constructor has any static methods which were previously stored on th.Tensor.
    Otherwise, the library might look for them there, not find them, and then trigger an error.

    Args:
        from_class (Type): the class on which we look for static methods co copy
        to_class (Type): the class onto which we copy all static methods found in <from_class>

    """
    # there are no static methods if from_class itself is not a type (sometimes funcs get passed in)

    for attr in dir(from_class):
        if is_static_method(klass=from_class, attr=attr):
            setattr(to_class, attr, getattr(from_class, attr))


def get_original_constructor_name(object_name: str) -> str:
    """Generate name for original constructor

    For each custom constructor, we move the original constructor to a consistent location relative to
    the original constructor so that each custom constructor automatically knows where to find the original
    method it is overloading. Namely, we move the original constructor to a different attr within the same
    module as the original constructor. This method specifies the naming convention that we use to name
    the original constructor when it is moved.

      Args:
          object_name (str): the original constructor's original name
    """

    return f"original_{object_name}"


def replace_classes_in_module(
    module: ModuleType,
    from_class: Callable,
    to_class: Callable,
    ignore_prefix: Optional[str] = None,
) -> None:
    """Recursively replace occurrence of `from_class` to `to_class` inside module.

    For example, when syft replaces torch.nn.parameter.Parameter constructor,
    there's also need to replace the same constructor in other modules that has already
    imported it.

    Args:
        module (ModuleType): top-level module to traverse
        from_class: Original constructor
        to_class: syft's ObjectConstructor
    """
    visited_modules = []

    def recursive_update(
        module: ModuleType, attr_name: Union[str, None] = None
    ) -> None:

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
    return f"{klass.__module__}.{klass.__qualname__}"


def full_name_with_name(klass: type) -> str:
    return f"{klass.__module__}.{klass.__name__}"
