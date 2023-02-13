# stdlib
from typing import Union

action_types = {}


def action_type_for_type(obj_or_type: Union[object, type]) -> type:
    if type(obj_or_type) == type:
        return action_types[obj_or_type]
    return action_types[type(obj_or_type)]
