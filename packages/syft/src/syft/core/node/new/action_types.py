# stdlib
from typing import Union

action_types = {}


def action_type_for_type(obj_or_type: Union[object, type]) -> type:
    if type(obj_or_type) != type:
        obj_or_type = type(obj_or_type)

    if obj_or_type not in action_types:
        return None
    return action_types[obj_or_type]
