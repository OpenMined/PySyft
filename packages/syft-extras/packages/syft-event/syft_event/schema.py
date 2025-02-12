from __future__ import annotations

import inspect
from inspect import signature

from pydantic import BaseModel
from typing_extensions import Any, Callable, Dict, Union, get_type_hints

from syft_event.types import Request, Response


def get_type_schema(type_hint: Any) -> Union[str, Dict[str, Any]]:
    """Get a schema representation of a type."""
    # Handle None
    if type_hint is None:
        return "null"

    # Handle Pydantic models
    if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
        return {
            "type": "model",
            "name": type_hint.__name__,
            "schema": type_hint.model_json_schema(),
        }

    # Handle Lists
    if getattr(type_hint, "__origin__", None) is list:
        return {"type": "array", "items": get_type_schema(type_hint.__args__[0])}

    # Handle Optional
    if getattr(type_hint, "__origin__", None) is Union:
        types = [t for t in type_hint.__args__ if t is not type(None)]
        if len(types) == 1:  # Optional[T] case
            return get_type_schema(types[0])
        return "union"  # General Union case

    # Handle basic types
    if isinstance(type_hint, type):
        return type_hint.__name__.lower()

    return "any"


def generate_schema(func: Callable) -> Dict[str, Any]:
    """Generate RPC schema from a function."""
    sig = signature(func)
    hints = get_type_hints(func)

    # Process parameters
    params = {}
    for name, param in sig.parameters.items():
        ptype = hints.get(name, Any)
        if inspect.isclass(ptype) and ptype is Request:
            continue
        params[name] = {
            "type": get_type_schema(ptype),
            "required": param.default is param.empty,
        }

    # Process return type
    ret_ptype = hints.get("return", Any)
    if inspect.isclass(ret_ptype) and ret_ptype is Response:
        # could be anything what the dev wants
        ret_ptype = Any

    return {
        "description": inspect.getdoc(func),
        "args": params,
        "returns": get_type_schema(ret_ptype),
    }
