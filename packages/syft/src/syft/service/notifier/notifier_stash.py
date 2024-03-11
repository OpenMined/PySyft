# stdlib

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_permissions import ActionObjectPermission
from .notifier import NotifierSettings

NamePartitionKey = PartitionKey(key="name", type_=str)
ActionIDsPartitionKey = PartitionKey(key="action_ids", type_=list[UID])


@instrument
@serializable()
class NotifierStash(BaseStash):
    object_type = NotifierSettings
    settings: PartitionSettings = PartitionSettings(
        name=NotifierSettings.__canonical_name__, object_type=NotifierSettings
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def admin_verify_key(self) -> SyftVerifyKey:
        return self.partition.root_verify_key

    # TODO: should this method behave like a singleton?
    def get(self, credentials: SyftVerifyKey) -> Result[NotifierSettings, Err]:
        """Get Settings"""
        result = self.get_all(credentials)
        if result.is_ok():
            settings = result.ok()
            if len(settings) == 0:
                return Ok(
                    None
                )  # TODO: Stash shouldn't be empty after init. Return Err instead?
            result = settings[
                0
            ]  # TODO: Should we check if theres more than one? => Report corruption
            return Ok(result)
        else:
            return Err(message=result.err())

    def set(
        self,
        credentials: SyftVerifyKey,
        settings: NotifierSettings,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
    ) -> Result[NotifierSettings, Err]:
        result = self.check_type(settings, self.object_type)
        # we dont use and_then logic here as it is hard because of the order of the arguments
        if result.is_err():
            return Err(message=result.err())
        return super().set(
            credentials=credentials, obj=result.ok()
        )  # TODO check if result isInstance(Ok)

    def update(
        self,
        credentials: SyftVerifyKey,
        settings: NotifierSettings,
        has_permission: bool = False,
    ) -> Result[NotifierSettings, Err]:
        result = self.check_type(settings, self.object_type)
        # we dont use and_then logic here as it is hard because of the order of the arguments
        if result.is_err():
            return Err(message=result.err())
        return super().update(
            credentials=credentials, obj=result.ok()
        )  # TODO check if result isInstance(Ok)
