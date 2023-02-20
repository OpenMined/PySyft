from typing import List
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Any
from enum import Enum
import hashlib
from RestrictedPython import compile_restricted
import inspect
from inspect import Parameter
from inspect import Signature
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.uid import UID
from .credentials import SyftVerifyKey
from .document_store import PartitionKey
from ...common.serde.serializable import serializable
from ...common.serde import _serialize
from .user_code import UserCode
from .transforms import transform
from .transforms import generate_id
from .transforms import TransformContext
from .response import SyftError

# TODO: check if we need 2 partition keys or if one is enough
UserVerifyKeyPartitionKey = PartitionKey(key="user_verify_key", type_=SyftVerifyKey)
UserVerifyKeyPartitionKey = PartitionKey(key="user_verify_key", type_=SyftVerifyKey)
PyCodeObject = Any

@serializable(recursive_serde=True)
class Policy(SyftObject):
    __canonical_name__ = "Policy"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    user_verify_key: SyftVerifyKey
    policy_code_uid: UID
    user_code_uid: UID
    serde: Optional[bytes] = None
    
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
    
# def create_object(context: TransformContext) -> TransformContext:
#     result = context.node.get_api().services.policy_code.get_by_uid(context.output["policy_code_uid"])
#     # TODO
#     exec(result.byte_code)
#     policy_name = eval(result.unique_name)
#     obj = policy_name(**context.output["policy_init_args"])
#     context.output["serde"] = _serialize(obj, to_bytes=True)
#     return context


@serializable(recursive_serde=True)
class PolicyCodeStatus(Enum):
    SUBMITTED = "submitted"
    DENIED = "denied"
    APPROVED = "approved"

@serializable(recursive_serde=True)
class PolicyCode(SyftObject):
    __canonical_name__ = "PolicyCode"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    user_verify_key: SyftVerifyKey
    raw_code: str
    parsed_code: str
    signature: inspect.Signature
    class_name: str 
    unique_name: str
    code_hash: str
    byte_code: PyCodeObject
    status: PolicyCodeStatus = PolicyCodeStatus.SUBMITTED
    
    
@serializable(recursive_serde=True)
class SubmitPolicyCode(SyftObject):
    __canonical_name__ = "SubmitPolicyCode"
    __version__ = SYFT_OBJECT_VERSION_1
    
    id: Optional[UID]
    code: str
    class_name: str
    input_kwargs: List[str]
    
    def compile(self) -> PyCodeObject:
        return compile_restricted(self.code, "<string>", "exec")

def hash_code(context: TransformContext) -> TransformContext:
    code = context.output["code"]
    del context.output["code"]
    context.output["raw_code"] = code
    code_hash = hashlib.sha256(code.encode("utf8")).hexdigest()
    context.output["code_hash"] = code_hash
    return context

def generate_unique_class_name(context: TransformContext) -> TransformContext:
    code_hash = context.output["code_hash"]
    service_class_name = context.output["class_name"]
    unique_name = f"user_func_{service_class_name}_{context.credentials}_{code_hash}"
    context.output["unique_name"] = unique_name
    return context

def check_class_code(context: TransformContext) -> TransformContext:
    # TODO: define the proper checking for this case based on the ideas from UserCode
    # check for no globals
    # check for Policy template -> __init__, apply_output, public_state
    # parse init signature
    # check dangerous libraries, maybe compile_restricted already does that
    parsed_code = context.output["raw_code"]
    context.output["parsed_code"] = parsed_code
    return context

def compile_code(context: TransformContext) -> TransformContext:
    # byte_code = compile_restricted(context.output["parsed_code"], "<string>", "exec")
    byte_code = compile(context.output["parsed_code"], "<string>", "exec")
    context.output["byte_code"] = byte_code
    return context

def add_credentials_for_key(key: str) -> Callable:
    def add_credentials(context: TransformContext) -> TransformContext:
        context.output[key] = context.credentials
        return context

    return add_credentials

def generate_signature(context: TransformContext) -> TransformContext:
    for k in context.output["input_kwargs"]:
        param = Parameter(name=k, kind=Parameter.POSITIONAL_OR_KEYWORD)
    sig = Signature(parameters=[param])
    context.output["signature"] = sig
    return context

@transform(CreatePolicy, Policy)
def create_policy_to_policy() -> List[Callable]:
    return [
        generate_id,
        add_credentials_for_key("user_verify_key")
        # check_policy_uid,
        # check_user_uid,
        # create_object
    ]

@transform(SubmitPolicyCode, PolicyCode)
def submit_policy_code_to_user_code() -> List[Callable]:
    return [
        generate_id,
        hash_code,
        generate_unique_class_name,
        generate_signature,
        check_class_code,
        compile_code,
        add_credentials_for_key("user_verify_key")
    ]