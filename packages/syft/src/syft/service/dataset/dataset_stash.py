# stdlib

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...types.uid import UID
from ...util.telemetry import instrument
from .dataset import Dataset


@instrument
@serializable(canonical_name="DatasetStashSQL", version=1)
class DatasetStash(ObjectStash[Dataset]):
    object_type = Dataset
    settings: PartitionSettings = PartitionSettings(
        name=Dataset.__canonical_name__, object_type=Dataset
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_name(
        self, credentials: SyftVerifyKey, name: str
    ) -> Result[Dataset | None, str]:
        return self.get_one_by_field(
            credentials=credentials, field_name="name", field_value=name
        )

    def search_action_ids(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[list[Dataset], str]:
        return self.get_all_by_field(
            credentials=credentials,
            field_name="action_ids",
            field_value=str(uid),
        )

    def get_all(
        self,
        credentials: SyftVerifyKey,
        order_by: PartitionKey | None = None,
        has_permission: bool = False,
    ) -> Ok[list] | Err[str]:
        result = self.get_all_by_field(
            credentials=credentials,
            field_name="to_be_deleted",
            field_value=False,
        )
        return result
