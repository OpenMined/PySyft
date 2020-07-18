from typing import List
from .decorators import type_hints


@type_hints
def get_subclasses(obj_type: type) -> List:
    classes = list()
    for sc in obj_type.__subclasses__():
        classes.append(sc)
        classes += get_subclasses(obj_type=sc)
    return classes
