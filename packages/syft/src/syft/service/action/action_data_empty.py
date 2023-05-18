# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Optional
from typing import Type

# relative
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject


@serializable()
class ActionDataEmpty(SyftObject):
    __canonical_name__ = "ActionDataEmpty"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_internal_type: Optional[Type] = Any

    def __str__(self) -> str:
        return f"{type(self).__name__}: {self.syft_internal_type.__name__}"
