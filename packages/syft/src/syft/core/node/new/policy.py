from typing import List
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Any

from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.uid import UID
from .credentials import SyftVerifyKey
from .document_store import PartitionKey
from ...common.serde.serializable import serializable
from ...common.serde import _serialize
from .policy_code import PolicyCode
from .user_code import UserCode
from .transforms import transform
from .transforms import generate_id
from .transforms import TransformContext
from .response import SyftError

UserVerifyKeyPartitionKey = PartitionKey(key="user_verify_key", type_=SyftVerifyKey)

@serializable(recursive_serde=True)
class Policy(SyftObject):
    __canonical_name__ = "Policy"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    user_verify_key: SyftVerifyKey
    policy_code_uid: UID
    user_code_uid: UID
    serde: bytes
    
@serializable(recursive_serde=True)
class CreatePolicy(SyftObject):
    __canonical_name__ = "CreatePolicy"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    policy_code_uid: UID
    user_code_uid: UID
    policy_init_args: Dict[str, Any]

def check_policy_uid(context: TransformContext) -> TransformContext:
    result = context.node.api.services.policy_code.get_by_uid(context.output["policy_code_uid"])
    if not result.is_ok():
        return SyftError(message=result.err())
    return context

def check_user_uid(context: TransformContext) -> TransformContext:
    result = context.node.api.services.user_code.get_by_uid(context.output["user_code_uid"])
    if not result.is_ok():
        return SyftError(message=result.err())
    return context
    
def create_object(context: TransformContext) -> TransformContext:
    print(context.node)
    print(dir(context))
    context.node.get_api().services.policy_code.get_all()
    result = context.node.get_api().services.policy_code.get_by_uid(context.output["policy_code_uid"])
    # TODO
    exec(result.byte_code)
    policy_name = eval(result.unique_name)
    obj = policy_name(**context.output["policy_init_args"])
    context.output["serde"] = _serialize(obj, to_bytes=True)
    return context

@transform(CreatePolicy, Policy)
def create_policy_to_policy() -> List[Callable]:
    return [
        generate_id,
        # check_policy_uid,
        # check_user_uid,
        create_object
    ]
