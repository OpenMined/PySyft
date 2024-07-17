# stdlib
from typing import Any

# relative
from ..serde.serializable import serializable
from .base import SyftBaseModel


@serializable(canonical_name="CachedSyftObject", version=1)
class CachedSyftObject(SyftBaseModel):
    """This class is used to represent the cached result."""

    result: Any
    error_msg: str | None = None
