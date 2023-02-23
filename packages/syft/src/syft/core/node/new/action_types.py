# stdlib
from typing import Any
from typing import Union

action_types = {}


def action_type_for_type(obj_or_type: Union[object, type]) -> type:
    if type(obj_or_type) != type:
        obj_or_type = type(obj_or_type)

    if obj_or_type not in action_types:
        print(f"WARNING: No Type for {obj_or_type}, returning {action_types[Any]}")
        return action_types[Any]
    return action_types[obj_or_type]
