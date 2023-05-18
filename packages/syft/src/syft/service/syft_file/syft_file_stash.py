# stdlib
from typing import List
from typing import Optional

# third party
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...types.uid import UID
from ...util.telemetry import instrument
from ...types.syft_file import SyftFile

FilenamePartitionKey = PartitionKey(key="filename", type_=str)
ActionIDsPartitionKey = PartitionKey(key="action_ids", type_=List[UID])


@instrument
@serializable()
class SyftFileStash(BaseUIDStoreStash):
    object_type = SyftFile
    settings: PartitionSettings = PartitionSettings(
        name=SyftFile.__canonical_name__, object_type=SyftFile
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_filename(
        self, credentials: SyftVerifyKey, filename: str
    ) -> Result[Optional[SyftFile], str]:
        qks = QueryKeys(qks=[FilenamePartitionKey.with_obj(filename)])
        return self.query_one(credentials=credentials, qks=qks)
