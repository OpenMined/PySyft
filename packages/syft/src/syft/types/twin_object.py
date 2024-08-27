# future
from __future__ import annotations

# stdlib
import logging
from typing import Any
from typing import ClassVar

# third party
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import Self

# relative
from ..client.client import SyftClient
from ..serde.serializable import serializable
from ..service.action.action_object import ActionObject
from ..service.action.action_object import TwinMode
from ..service.action.action_types import action_types
from ..service.response import SyftSuccess
from ..service.response import SyftWarning
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from .errors import SyftException
from .result import as_result
from .syft_object import SyftObject
from .uid import UID

logger = logging.getLogger(__name__)


def to_action_object(obj: Any) -> ActionObject:
    if isinstance(obj, ActionObject):
        return obj

    if type(obj) in action_types:
        return action_types[type(obj)](syft_action_data_cache=obj)
    raise ValueError(f"{type(obj)} not in action_types")


@serializable()
class TwinObject(SyftObject):
    __canonical_name__ = "TwinObject"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_searchable__: ClassVar[list[str]] = []

    id: UID
    private_obj: ActionObject
    private_obj_id: UID = None  # type: ignore
    mock_obj: ActionObject
    mock_obj_id: UID = None  # type: ignore

    @field_validator("private_obj", mode="before")
    @classmethod
    def make_private_obj(cls, v: Any) -> ActionObject:
        return to_action_object(v)

    @model_validator(mode="after")
    def make_private_obj_id(self) -> Self:
        if self.private_obj_id is None:
            self.private_obj_id = self.private_obj.id  # type: ignore[unreachable]
        return self

    @field_validator("mock_obj", mode="before")
    @classmethod
    def make_mock_obj(cls, v: Any) -> ActionObject:
        return to_action_object(v)

    @model_validator(mode="after")
    def make_mock_obj_id(self) -> Self:
        if self.mock_obj_id is None:
            self.mock_obj_id = self.mock_obj.id  # type: ignore[unreachable]
        return self

    @property
    def private(self) -> ActionObject:
        twin_id = self.id
        private = self.private_obj
        private.syft_twin_type = TwinMode.PRIVATE
        private.id = twin_id
        return private

    @property
    def mock(self) -> ActionObject:
        twin_id = self.id
        mock = self.mock_obj
        mock.syft_twin_type = TwinMode.MOCK
        mock.id = twin_id
        return mock

    @as_result(SyftException)
    def _save_to_blob_storage(
        self, allow_empty: bool = False
    ) -> SyftSuccess | SyftWarning:
        # Set server location and verify key
        self.private_obj._set_obj_location_(
            self.syft_server_location,
            self.syft_client_verify_key,
        )
        self.mock_obj._set_obj_location_(
            self.syft_server_location,
            self.syft_client_verify_key,
        )
        self.mock_obj._save_to_blob_storage(allow_empty=allow_empty).unwrap()
        return self.private_obj._save_to_blob_storage(allow_empty=allow_empty).unwrap()

    def send(self, client: SyftClient, add_storage_permission: bool = True) -> Any:
        self._set_obj_location_(client.id, client.verify_key)
        blob_store_result = self._save_to_blob_storage().unwrap()
        if isinstance(blob_store_result, SyftWarning):
            logger.debug(blob_store_result.message)
        res = client.api.services.action.set(
            self,
            add_storage_permission=add_storage_permission,
        )
        return res
