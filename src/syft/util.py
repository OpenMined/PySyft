from typing import List

# breaking convention here because index_globals needs
# the full syft name to be present.
import syft  # noqa: F401

from .decorators import type_hints


@type_hints
def get_subclasses(obj_type: type) -> List:
    classes = list()
    for sc in obj_type.__subclasses__():
        classes.append(sc)
        classes += get_subclasses(obj_type=sc)
    return classes


def index_modules(a_dict, keys):
    if len(keys) == 0:
        return a_dict
    return index_modules(a_dict.__dict__[keys[0]], keys[1:])


def index_syft_by_module_name(fully_qualified_name):
    attr_list = fully_qualified_name.split(".")
    assert attr_list[0] == "syft"
    assert attr_list[1] == "core"
    return index_modules(globals()["syft"], attr_list[1:])


def get_fully_qualified_name(obj):
    fqn = obj.__module__
    try:
        fqn += "." + obj.__name__
    except Exception as e:
        print(f"Failed to get FQN: {e}")
    return fqn
