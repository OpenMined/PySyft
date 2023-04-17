# stdlib
from typing import Any
from typing import Union

# relative
from .action_data_empty import ActionDataEmpty

action_types = {}


def action_type_for_type(obj_or_type: Union[object, type]) -> type:
    if type(obj_or_type) != type:
        if isinstance(obj_or_type, ActionDataEmpty):
            obj_or_type = obj_or_type.syft_internal_type
        else:
            obj_or_type = type(obj_or_type)

    if obj_or_type not in action_types:
        # 🟡 TODO: print(f"WARNING: No Type for {obj_or_type}, returning {action_types[Any]}")
        return action_types[Any]
    return action_types[obj_or_type]
