from enum import Enum
import hashlib
import inspect
from inspect import Parameter
from inspect import Signature
from typing import Optional
from typing import List
from typing import Dict
from typing import Any
from typing import Callable
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.uid import UID
from .credentials import SyftVerifyKey
from .document_store import PartitionKey
from .credentials import SyftVerifyKey
from ...common.serde.serializable import serializable
from .transforms import generate_id
from .transforms import TransformContext
from .transforms import transform
from RestrictedPython import compile_restricted

UserVerifyKeyPartitionKey = PartitionKey(key="user_verify_key", type_=SyftVerifyKey)

PyCodeObject = Any

# TODO: think about logging code runs
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
    