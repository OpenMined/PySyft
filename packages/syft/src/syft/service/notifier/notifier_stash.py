# stdlib

# third party

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import DocumentStore
from ...store.document_store import NewBaseStash
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.result import as_result
from ...types.uid import UID
from ..action.action_permissions import ActionObjectPermission
from .notifier import NotifierSettings

NamePartitionKey = PartitionKey(key="name", type_=str)
ActionIDsPartitionKey = PartitionKey(key="action_ids", type_=list[UID])


@serializable(canonical_name="NotifierStash", version=1)
class NotifierStash(NewBaseStash):
    object_type = NotifierSettings
    settings: PartitionSettings = PartitionSettings(
        name=NotifierSettings.__canonical_name__, object_type=NotifierSettings
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def admin_verify_key(self) -> SyftVerifyKey:
        return self.partition.root_verify_key

    # TODO: should this method behave like a singleton?
    @as_result(StashException, NotFoundException)
    def get(self, credentials: SyftVerifyKey) -> NotifierSettings:
        """Get Settings"""
        settings: list[NotifierSettings] = self.get_all(credentials).unwrap()
        if len(settings) == 0:
            raise NotFoundException
        return settings[0]

    @as_result(StashException)
    def set(
        self,
        credentials: SyftVerifyKey,
        settings: NotifierSettings,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
    ) -> NotifierSettings:
        result = self.check_type(settings, self.object_type).unwrap()
        # we dont use and_then logic here as it is hard because of the order of the arguments
        return (
            super().set(credentials=credentials, obj=result).unwrap()
        )  # TODO check if result isInstance(Ok)
