""" A set of useful methods used by the syft.core.node.common.node_table submodule. """
# stdlib
from typing import Any
from typing import Dict

# relative
from .user import SyftObject

# attribute names representing a date owned by the PyGrid's database schemas.
datetime_cols = ["date", "created_at", "destroyed_at", "deployed_on", "updated_on"]


def model_to_json(model: Any) -> Dict[str, Any]:
    """
    Returns a JSON representation of an SQLAlchemy-backed object.

    Args:
        model: SQLAlchemy-backed object to be represented as a JSON data structure.
    Returns:
        Dict: Python dictionary representing the SQLAlchemy object.
    """
    json = {}
    for col in model.__mapper__.attrs.keys():  # type: ignore
        if col != "hashed_password" and col != "salt":
            if col in datetime_cols:
                # Cast datetime object to string
                json[col] = str(getattr(model, col))
            else:
                json[col] = getattr(model, col)
    return json


def syft_object_to_json(obj: SyftObject) -> Dict[str, Any]:
    """
    Returns a JSON representation of an NoSQL-backed object.

    Args:
        model: NoSQL Syft Object document
    Returns:
        Dict: Python dictionary representing the NoSQL object.
    """
    json = {}
    for field in obj.__attr_state__:
        if field != "hashed_password" and field != "salt":
            if field in datetime_cols:
                json[field] = str(getattr(obj, field))
            else:
                json[field] = getattr(obj, field)

    return json
