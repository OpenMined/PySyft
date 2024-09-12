# stdlib
from enum import Enum
import json
from typing import Any
from uuid import UUID

# relative
from ...serde.json_serde import Json
from ...types.uid import UID


def _default_dumps(val: Any) -> Json:  # type: ignore
    if isinstance(val, UID):
        return str(val.no_dash)
    elif isinstance(val, UUID):
        return val.hex
    elif issubclass(type(val), Enum):
        return val.name
    elif val is None:
        return None
    return str(val)


def _default_loads(val: Any) -> Any:  # type: ignore
    if "UID" in val:
        return UID(val)
    return val


def dumps(d: Any) -> str:
    return json.dumps(d, default=_default_dumps)


def loads(d: str) -> Any:
    return json.loads(d, object_hook=_default_loads)
