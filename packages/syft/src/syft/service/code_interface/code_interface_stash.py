# stdlib
from typing import List
from typing import Optional

# third party
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ..action.action_permissions import ActionObjectPermission
from ..code.user_code import UserCode
from .code_interface import CodeInterface

NamePartitionKey = PartitionKey(key="name", type_=str)


class CodeInterfaceStash(BaseUIDStoreStash):
    object_type = CodeInterface
    settings: PartitionSettings = PartitionSettings(
        name=CodeInterface.__canonical_name__, object_type=CodeInterface
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def set(
        self,
        credentials: SyftVerifyKey,
        code_interface: CodeInterface,
        code: UserCode,
        add_permissions: Optional[List[ActionObjectPermission]] = None,
    ) -> Result[CodeInterface, str]:
        res = self.check_type(code_interface, self.object_type)
        if res.is_err():
            return res

        # Add the new code to the user_code_mapping
        code_interface.add_code(code)

        return super().set(
            credentials=credentials, obj=res.ok(), add_permissions=add_permissions
        )