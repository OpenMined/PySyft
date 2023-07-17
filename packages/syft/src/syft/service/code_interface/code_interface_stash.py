# stdlib
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
from .code_interface import CodeInterface

NamePartitionKey = PartitionKey(key="service_func_name", type_=str)


@serializable()
class CodeInterfaceStash(BaseUIDStoreStash):
    object_type = CodeInterface
    settings: PartitionSettings = PartitionSettings(
        name=CodeInterface.__canonical_name__, object_type=CodeInterface
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_service_func_name(
        self, credentials: SyftVerifyKey, service_func_name: str
    ) -> Result[Optional[CodeInterface], str]:
        qks = QueryKeys(qks=[NamePartitionKey.with_obj(service_func_name)])
        return self.query_one(credentials=credentials, qks=qks)

    # def set(
    #     self,
    #     credentials: SyftVerifyKey,
    #     code_interface: CodeInterface,
    #     code: UserCode,
    #     add_permissions: Optional[List[ActionObjectPermission]] = None,
    # ) -> Result[CodeInterface, str]:
    #     res = self.check_type(code_interface, self.object_type)
    #     if res.is_err():
    #         return res

    #     # Add the new code to the user_code_mapping
    #     code_interface.add_code(code)

    #     return super().set(
    #         credentials=credentials, obj=res.ok(), add_permissions=add_permissions
    #     )

    # def get_version(self, name:str, version:int) -> Optional[UserCode]:
    #     for obj in self.objs.values():
    #         if obj.name == name and obj.version == version:
    #             return obj
    #     return None
