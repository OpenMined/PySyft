from typing import List
from .decorators.syft_decorator_impl import syft_decorator

# breaking convention here because index_globals needs
# the full syft name to be present.
import syft  # noqa: F401


@syft_decorator(typechecking=True)
def get_subclasses(obj_type: type) -> List:
    classes = list()
    for sc in obj_type.__subclasses__():
        classes.append(sc)
        classes += get_subclasses(obj_type=sc)
    return classes


@syft_decorator(typechecking=True)
def index_modules(a_dict, keys):
    if len(keys) == 0:
        return a_dict
    return index_modules(a_dict.__dict__[keys[0]], keys[1:])


@syft_decorator(typechecking=True)
def index_syft_by_module_name(fully_qualified_name:str):
    attr_list = fully_qualified_name.split(".")
    assert attr_list[0] == "syft"
    assert attr_list[1] == "core"
    return index_modules(globals()["syft"], attr_list[1:])


@syft_decorator(typechecking=True)
def get_fully_qualified_name(obj: object) -> str:
    """Return the full path and name of a class

    Sometimes we want to return the entire path and name encoded
    using periods. For example syft.core.common.message.SyftMessage
    is the current fully qualified path and name for the SyftMessage
    object."""

    fqn = obj.__module__
    try:
        fqn += "." + obj.__name__
    except Exception as e:
        print(f"Failed to get FQN: {e}")
    return fqn
