# stdlib
from typing import Any
from typing import Type

# relative
from ...util.logger import debug
from .action_data_empty import ActionDataEmpty

action_types = {}


def action_type_for_type(obj_or_type: Any) -> Type:
    """Convert standard type to Syft types

    Parameters:
        obj_or_type: Union[object, type]
            Can be an object or a class
    """
    if type(obj_or_type) != type:
        if isinstance(obj_or_type, ActionDataEmpty):
            obj_or_type = obj_or_type.syft_internal_type
        else:
            obj_or_type = type(obj_or_type)

    if obj_or_type not in action_types:
        debug(f"WARNING: No Type for {obj_or_type}, returning {action_types[Any]}")
        return action_types[Any]

    return action_types[obj_or_type]


def action_type_for_object(obj: Any) -> Type:
    """Convert standard type to Syft types

    Parameters:
        obj_or_type: Union[object, type]
            Can be an object or a class
    """
    _type = type(obj)

    if _type not in action_types:
        debug(f"WARNING: No Type for {_type}, returning {action_types[Any]}")
        return action_types[Any]

    return action_types[_type]
