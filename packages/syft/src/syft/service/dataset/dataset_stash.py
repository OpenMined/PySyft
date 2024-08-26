# stdlib

# stdlib

# third party
from result import Result

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from ...types.uid import UID
from ...util.telemetry import instrument
from .dataset import Dataset


@instrument
@serializable(canonical_name="DatasetStashSQL", version=1)
class DatasetStash(ObjectStash[Dataset]):
    settings: PartitionSettings = PartitionSettings(
        name=Dataset.__canonical_name__, object_type=Dataset
    )

    def get_by_name(
        self, credentials: SyftVerifyKey, name: str
    ) -> Result[Dataset | None, str]:
        return self.get_one_by_field(
            credentials=credentials, field_name="name", field_value=name
        )

    def search_action_ids(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[list[Dataset], str]:
        return self.get_all_contains(
            credentials=credentials,
            field_name="action_ids",
            field_value=uid.no_dash,
        )

    def get_all(
        self,
        credentials: SyftVerifyKey,
        has_permission: bool = False,
        order_by: str | None = None,
        sort_order: str = "asc",
        limit: int | None = None,
        offset: int | None = None,
    ) -> Result[list[Dataset], str]:
        result = self.get_all_by_field(
            credentials=credentials,
            field_name="to_be_deleted",
            field_value=False,
            order_by=order_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset,
        )
        return result
