# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Optional

# relative
from ..serde.serializable import serializable
from ..service.action.action_object import ActionObject
from ..service.action.action_types import action_types
from .syft_object import SyftObject
from .uid import UID


def to_action_object(obj: Any) -> ActionObject:
    if isinstance(obj, ActionObject):
        return obj

    if type(obj) in action_types:
        return action_types[type(obj)](syft_action_data=obj)
    raise Exception(f"{type(obj)} not in action_types")


@serializable()
class TwinObject(SyftObject):
    __canonical_name__ = "TwinObject"
    __version__ = 1

    __attr_searchable__ = []

    private_obj: ActionObject
    private_obj_id: UID
    mock_obj: ActionObject
    mock_obj_id: UID

    def __init__(
        self,
        private_obj: ActionObject,
        mock_obj: ActionObject,
        private_obj_id: Optional[UID] = None,
        mock_obj_id: Optional[UID] = None,
        id: Optional[UID] = None,
    ) -> None:
        private_obj = to_action_object(private_obj)
        mock_obj = to_action_object(mock_obj)

        if private_obj_id is None:
            private_obj_id = private_obj.id
        if mock_obj_id is None:
            mock_obj_id = mock_obj.id
        if id is None:
            id = UID()
        super().__init__(
            private_obj=private_obj,
            private_obj_id=private_obj_id,
            mock_obj=mock_obj,
            mock_obj_id=mock_obj_id,
            id=id,
        )

    @property
    def private(self) -> ActionObject:
        twin_id = self.id
        private = self.private_obj
        private.id = twin_id
        return private

    @property
    def mock(self) -> ActionObject:
        twin_id = self.id
        mock = self.mock_obj
        mock.id = twin_id
        return mock
