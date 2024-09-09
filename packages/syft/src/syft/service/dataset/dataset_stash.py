# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.query import Filter
from ...store.db.stash import ObjectStash
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.result import as_result
from ...types.uid import UID
from ...util.trace_decorator import instrument
from .dataset import Dataset


@instrument
@serializable(canonical_name="DatasetStashSQL", version=1)
class DatasetStash(ObjectStash[Dataset]):
    @as_result(StashException, NotFoundException)
    def get_by_name(self, credentials: SyftVerifyKey, name: str) -> Dataset:
        return self.get_one(credentials=credentials, filters={"name": name}).unwrap()

    @as_result(StashException)
    def search_action_ids(self, credentials: SyftVerifyKey, uid: UID) -> list[Dataset]:
        return self.get_all(
            credentials=credentials,
            filters={"action_ids": Filter("action_ids", "contains", uid)},
        ).unwrap()

    @as_result(StashException)
    def get_all(
        self,
        credentials: SyftVerifyKey,
        has_permission: bool = False,
        order_by: str | None = None,
        sort_order: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Dataset]:
        # TODO standardize soft delete and move to ObjectStash.get_all
        return (
            super()
            .get_all(
                credentials=credentials,
                filters={"to_be_deleted": False},
                has_permission=has_permission,
                order_by=order_by,
                sort_order=sort_order,
                limit=limit,
                offset=offset,
            )
            .unwrap()
        )
