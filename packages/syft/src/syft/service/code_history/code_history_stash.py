# stdlib
from typing import List

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
from .code_history import CodeHistory

NamePartitionKey = PartitionKey(key="service_func_name", type_=str)


@serializable()
class CodeHistoryStash(BaseUIDStoreStash):
    object_type = CodeHistory
    settings: PartitionSettings = PartitionSettings(
        name=CodeHistory.__canonical_name__, object_type=CodeHistory
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_service_func_name(
        self, credentials: SyftVerifyKey, service_func_name: str
    ) -> Result[List[CodeHistory], str]:
        qks = QueryKeys(qks=[NamePartitionKey.with_obj(service_func_name)])
        return self.query_all(credentials=credentials, qks=qks)

    def get_by_verify_key(
        self, credentials: SyftVerifyKey
    ) -> Result[List[CodeHistory], str]:
        return self.find_all(credentials=credentials)

    # def get_version(self, name:str, version:int) -> Optional[UserCode]:
    #     for obj in self.objs.values():
    #         if obj.name == name and obj.version == version:
    #             return obj
    #     return None