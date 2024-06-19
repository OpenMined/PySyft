# stdlib
import logging
from typing import Any

# relative
from .action_data_empty import ActionDataEmpty

logger = logging.getLogger(__name__)

action_types: dict = {}


def action_type_for_type(obj_or_type: Any) -> type:
    """Convert standard type to Syft types

    Parameters:
        obj_or_type: Union[object, type]
            Can be an object or a class
    """
    if isinstance(obj_or_type, ActionDataEmpty):
        obj_or_type = obj_or_type.syft_internal_type
    if type(obj_or_type) != type:
        obj_or_type = type(obj_or_type)

    if obj_or_type not in action_types:
        logger.debug(
            f"WARNING: No Type for {obj_or_type}, returning {action_types[Any]}"
        )

    return action_types.get(obj_or_type, action_types[Any])


def action_type_for_object(obj: Any) -> type:
    """Convert standard type to Syft types

    Parameters:
        obj_or_type: Union[object, type]
            Can be an object or a class
    """
    _type = type(obj)

    if _type not in action_types:
        logger.debug(f"WARNING: No Type for {_type}, returning {action_types[Any]}")
        return action_types[Any]

    return action_types[_type]
