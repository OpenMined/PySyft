# stdlib
from enum import Enum
import hashlib
from typing import Callable
from typing import List
from typing import Optional

# relative
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .credentials import SyftVerifyKey
from .document_store import CollectionKey
from .transforms import TransformContext
from .transforms import generate_id
from .transforms import transform

UserVerifyKeyCollectionKey = CollectionKey(key="user_verify_key", type_=SyftVerifyKey)
CodeHashCollectionKey = CollectionKey(key="code_hash", type_=int)


class UserCodeStatus(Enum):
    SUBMITTED = "submitted"
    DENIED = "denied"
    APPROVED = "approved"


@serializable(recursive_serde=True)
class UserCode(SyftObject):
    # version
    __canonical_name__ = "UserCode"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    user_verify_key: SyftVerifyKey
    code: str
    func_name: str
    code_hash: str
    status: UserCodeStatus = UserCodeStatus.SUBMITTED

    __attr_searchable__ = ["status", "func_name"]
    __attr_unique__ = ["user_verify_key", "code_hash"]


@serializable(recursive_serde=True)
class SubmitUserCode(SyftObject):
    # version
    __canonical_name__ = "SubmitUserCode"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]

    code: str
    func_name: str


def check_code(context: TransformContext) -> TransformContext:
    return context


def hash_code(context: TransformContext) -> TransformContext:
    code = context.output["code"]
    code_hash = hashlib.sha256(code.encode("utf8")).hexdigest()
    context.output["code_hash"] = code_hash

    return context


def add_credentials_for_key(key: str) -> Callable:
    def add_credentials(context: TransformContext) -> TransformContext:
        context.output[key] = context.credentials
        return context

    return add_credentials


@transform(SubmitUserCode, UserCode)
def submit_user_code_to_user_code() -> List[Callable]:
    return [
        generate_id,
        check_code,
        hash_code,
        add_credentials_for_key("user_verify_key"),
    ]
