# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.result import as_result
from ...types.uid import UID
from ...util.telemetry import instrument
from .dataset import Dataset


@instrument
@serializable(canonical_name="DatasetStashSQL", version=1)
class DatasetStash(ObjectStash[Dataset]):
    @as_result(StashException, NotFoundException)
    def get_by_name(self, credentials: SyftVerifyKey, name: str) -> Dataset:
        return self.get_one_by_field(
            credentials=credentials, field_name="name", field_value=name
        ).unwrap()

    @as_result(StashException)
    def search_action_ids(self, credentials: SyftVerifyKey, uid: UID) -> list[Dataset]:
        return self.get_all_contains(
            credentials=credentials,
            field_name="action_ids",
            field_value=uid.no_dash,
        ).unwrap()

    @as_result(StashException)
    def get_all(
        self,
        credentials: SyftVerifyKey,
        has_permission: bool = False,
        order_by: str | None = None,
        sort_order: str = "asc",
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Dataset]:
        # TODO standardize soft delete and move to ObjectStash.get_all
        return self.get_all_by_field(
            credentials=credentials,
            has_permission=has_permission,
            field_name="to_be_deleted",
            field_value=False,
            order_by=order_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset,
        ).unwrap()
