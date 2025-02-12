from __future__ import annotations

import inspect
import json
from dataclasses import is_dataclass

from pydantic import BaseModel
from syft_rpc.protocol import SyftRequest
from typing_extensions import Any, Callable, Dict, get_type_hints

from syft_event.types import Request


def func_args_from_request(func: Callable, request: SyftRequest) -> Dict[str, Any]:
    """Extract dependencies based on function type hints"""

    type_hints = get_type_hints(func)
    sig = inspect.signature(func)
    kwargs = {}

    for pname, _ in sig.parameters.items():
        ptype = type_hints.get(pname, Any)

        if inspect.isclass(ptype) and ptype is Request:
            kwargs[pname] = Request(
                id=str(request.id),
                sender=request.sender,
                url=request.url,
                headers=request.headers,
                body=request.body,
            )
        elif is_dataclass(ptype):
            kwargs[pname] = ptype(**request.json())
        elif inspect.isclass(ptype) and issubclass(ptype, BaseModel):
            kwargs[pname] = request.model(ptype)
        elif ptype is dict:
            val = json.loads(request.body.decode()) if request.body else None
            kwargs[pname] = val
        elif ptype is str:
            # Default to injecting body for unknown types
            kwargs[pname] = request.text()
        else:
            raise ValueError(f"Unknown type {ptype} for parameter {pname}")

    return kwargs
