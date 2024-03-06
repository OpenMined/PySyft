# future
from __future__ import annotations

# stdlib
from types import NoneType

# relative
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID


@serializable()
class ActionDataEmpty(SyftObject):
    __canonical_name__ = "ActionDataEmpty"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_internal_type: type | None = NoneType  # type: ignore

    def __repr__(self) -> str:
        return f"{type(self).__name__} <{self.syft_internal_type}>"

    def __str__(self) -> str:
        return f"{type(self).__name__} <{self.syft_internal_type}>"


@serializable()
class ObjectNotReady(SyftObject):
    __canonical_name__ = "ObjectNotReady"
    __version__ = SYFT_OBJECT_VERSION_1

    obj_id: UID


@serializable()
class ActionDataLink(SyftObject):
    __canonical_name__ = "ActionDataLink"
    __version__ = SYFT_OBJECT_VERSION_1

    action_object_id: UID
