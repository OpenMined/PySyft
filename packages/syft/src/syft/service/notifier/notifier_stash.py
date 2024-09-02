# stdlib

# third party

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.result import as_result
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_permissions import ActionObjectPermission
from .notifier import NotifierSettings

NamePartitionKey = PartitionKey(key="name", type_=str)
ActionIDsPartitionKey = PartitionKey(key="action_ids", type_=list[UID])


@instrument
@serializable(canonical_name="NotifierSQLStash", version=1)
class NotifierStash(ObjectStash[NotifierSettings]):
    settings: PartitionSettings = PartitionSettings(
        name=NotifierSettings.__canonical_name__, object_type=NotifierSettings
    )

    @as_result(StashException, NotFoundException)
    def get(self, credentials: SyftVerifyKey) -> NotifierSettings:
        """Get Settings"""
        # actually get latest settings
        result = self.get_all(credentials, limit=1, sort_order="desc").unwrap()
        if len(result) > 0:
            return result[0]
        raise NotFoundException(
            public_message="No settings found for the current user."
        )

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
        return super().set(credentials=credentials, obj=result).unwrap()
