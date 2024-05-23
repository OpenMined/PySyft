# stdlib

# stdlib
from typing import Any

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_permissions import ActionObjectPermission
from .settings import NodeSettings

NamePartitionKey = PartitionKey(key="name", type_=str)
ActionIDsPartitionKey = PartitionKey(key="action_ids", type_=list[UID])


@instrument
@serializable()
class SettingsStash(BaseUIDStoreStash):
    object_type = NodeSettings
    settings: PartitionSettings = PartitionSettings(
        name=NodeSettings.__canonical_name__, object_type=NodeSettings
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def check_type(self, obj: Any) -> Result[NodeSettings, str]:
        if isinstance(obj, NodeSettings):
            return Ok(obj)
        else:
            return Err(f"{type(obj)} does not match required type: {NodeSettings}")

    def get(self, credentials: SyftVerifyKey) -> Result[NodeSettings | None, str]:
        result = self.get_all(credentials=credentials)
        match result:
            case Ok(settings) if settings:
                return Ok(None)
            case Ok(settings):
                return Ok(settings[0])  # type: ignore
            case Err(err_message):
                return Err(err_message)

    def set(
        self,
        credentials: SyftVerifyKey,
        settings: NodeSettings,
        add_permission: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
    ) -> Result[NodeSettings, str]:
        result = self.check_type(settings)
        # we dont use and_then logic here as it is hard because of the order of the arguments
        match result:
            case Ok(obj):
                return super().set(credentials=credentials, obj=obj)  # type: ignore
            case Err(error):
                return Err(error)

    def update(
        self,
        credentials: SyftVerifyKey,
        settings: NodeSettings,
        has_permission: bool = False,
    ) -> Result[NodeSettings, str]:
        result = self.check_type(settings)
        # we dont use and_then logic here as it is hard because of the order of the arguments

        match result:
            case Ok(obj):
                return super().update(credentials=credentials, obj=obj)  # type: ignore
            case Err(error):
                return Err(error)
