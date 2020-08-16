from typing import List

# breaking convention here because index_globals needs
# the full syft name to be present.
import syft  # noqa: F401
from forbiddenfruit import curse

from .decorators.syft_decorator_impl import syft_decorator


@syft_decorator(typechecking=True)
def get_subclasses(obj_type: type) -> List[type]:
    """Recusively generate the list of all classes within the sub-tree of an object

    As a paradigm in Syft, we often allow for something to be known about by another
    part of the codebase merely because it has subclassed a particular object. While
    this can be a big "magicish" it also can simplify future extensions and reduce
    the likelihood of small mistakes (if done right).

    This is a utility function which allows us to look for sub-classes and the sub-classes
    of those sub-classes etc. returning a full list of descendants of a class in the inheritance
    hierarchy.

    :param obj_type: the type we want to look for sub-classes of
    :type type:
    :returns: the list of subclasses of obj_type:
    :rtype: List[type]

    """

    classes = list()
    for sc in obj_type.__subclasses__():
        classes.append(sc)
        classes += get_subclasses(obj_type=sc)
    return classes


@syft_decorator(typechecking=True)
def index_modules(a_dict: object, keys: List[str]) -> object:
    """Recursively find a syft module from its path

    This is the recursive inner function of index_syft_by_module_name.
    See that method for a full description.

    :param a_dict: a module we're traversing
    :type a_dict: object
    :param keys: the list of string attributes we're using to traverse the module
    :type keys: List[str]
    :returns: a reference to the final object
    :rtype: object

    """

    if len(keys) == 0:
        return a_dict
    return index_modules(a_dict=a_dict.__dict__[keys[0]], keys=keys[1:])


@syft_decorator(typechecking=True)
def index_syft_by_module_name(fully_qualified_name: str) -> object:
    """Look up a Syft class/module/function from full path and name

    Sometimes we want to use the fully qualified name (such as one
    generated from the 'get_fully_qualified_name' method below) to
    fetch an actual reference. This is most commonly used in deserialization
    so that we can have generic protobuf objects which just have a string
    representation of the specific object it is meant to deserialize to.

    :param fully_qualified_name: the name of a module, class, or function
    :type fully_qualified_name: str
    :returns: a reference to the actual object at that string path
    :rtype: object

    """

    attr_list = fully_qualified_name.split(".")
    assert attr_list[0] == "syft"
    assert attr_list[1] == "core" or attr_list[1] == "lib"
    return index_modules(a_dict=globals()["syft"], keys=attr_list[1:])


@syft_decorator(typechecking=True)
def get_fully_qualified_name(obj: object) -> str:
    """Return the full path and name of a class

    Sometimes we want to return the entire path and name encoded
    using periods. For example syft.core.common.message.SyftMessage
    is the current fully qualified path and name for the SyftMessage
    object.

    :param obj: the object we we want to get the name of
    :type obj: object
    :returns: the full path and name of the object
    :rtype: str

    """

    fqn = obj.__module__
    try:
        fqn += "." + obj.__class__.__name__
    except Exception as e:
        print(f"Failed to get FQN: {e}")
    return fqn


@syft_decorator(typechecking=True)
def aggressive_set_attr(obj: object, name: str, attr: object) -> None:
    """Different objects prefer different types of monkeypatching - try them all"""

    try:
        setattr(obj, name, attr)
    except Exception:
        curse(obj, name, attr)


def obj2pointer_type(obj):
    fqn = get_fully_qualified_name(obj=obj)
    ref = syft.lib_ast(fqn, return_callable=True)
    return ref.pointer_type