from typing import List
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Any

from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.uid import UID
from .credentials import SyftVerifyKey
from ...common.serde.serializable import serializable
from .policy_code import PolicyCode
from .user_code import UserCode
from .transforms import transform
from .transforms import generate_id
from .transforms import TransformContext
from .response import SyftError


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
    user_verify_key: SyftVerifyKey
    policy_code_uid: UID
    user_code_uid: UID
    policy_init_args: Dict[str, Any]

def check_policy_uid(context: TransformContext) -> TransformContext:
    result = context.node.api.services.policy_code.get_by_uid(context["policy_code_uid"])
    if not result.is_ok():
        return SyftError(message=result.err())
    return context

def check_user_uid(context: TransformContext) -> TransformContext:
    result = context.node.api.services.user_code.get_by_uid(context["user_code_uid"])
    if not result.is_ok():
        return SyftError(message=result.err())
    return context
    
def create_object(context: TransformContext) -> TransformContext:
    result = context.node.api.services.policy_code.get_by_uid(context["policy_code_uid"])
    # TODO
    object = result.value[""]
    context["serde"] = serialize(object, to_bytes=True)
    return context

@transform(CreatePolicy, Policy)
def create_policy_to_policy() -> List[Callable]:
    return [
        generate_id,
        check_policy_uid,
        check_user_uid,
        create_object
    ]
