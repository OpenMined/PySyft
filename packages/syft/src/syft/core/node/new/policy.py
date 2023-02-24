from typing import List
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Any
from typing import Type
from typing import Union
from result import Err
from result import Ok
from result import Result

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
from ...common.serde import _deserialize
from .transforms import transform
from .transforms import generate_id
from .transforms import TransformContext
from .response import SyftError
from .dataset import Asset
from .response import SyftError
from .response import SyftSuccess

UserVerifyKeyPartitionKey = PartitionKey(key="user_verify_key", type_=SyftVerifyKey)
PyCodeObject = Any

# @serializable(recursive_serde=True)
# class Policy(SyftObject):
#     __canonical_name__ = "Policy"
#     __version__ = SYFT_OBJECT_VERSION_1

#     id: UID
#     user_verify_key: SyftVerifyKey
#     policy_code_uid: UID
#     user_code_uid: UID
#     serde: Optional[bytes] = None
    
# @serializable(recursive_serde=True)
# class CreatePolicy(SyftObject):
#     __canonical_name__ = "CreatePolicy"
#     __version__ = SYFT_OBJECT_VERSION_1

#     id: Optional[UID]
#     policy_code_uid: UID
#     user_code_uid: UID
#     policy_init_args: Dict[str, Any]

# def check_policy_uid(context: TransformContext) -> TransformContext:
#     result = context.node.api.services.policy_code.get_by_uid(context.output["policy_code_uid"])
#     if not result.is_ok():
#         return SyftError(message=result.err())
#     return context

# def check_user_uid(context: TransformContext) -> TransformContext:
#     result = context.node.api.services.user_code.get_by_uid(context.output["user_code_uid"])
#     if not result.is_ok():
#         return SyftError(message=result.err())
#     return context
    
# def create_object(context: TransformContext) -> TransformContext:
#     result = context.node.get_api().services.policy_code.get_by_uid(context.output["policy_code_uid"])
#     # TODO
#     exec(result.byte_code)
#     policy_name = eval(result.unique_name)
#     obj = policy_name(**context.output["policy_init_args"])
#     context.output["serde"] = _serialize(obj, to_bytes=True)
#     return context

def extract_uids(kwargs: Dict[str, Any]) -> Dict[str, UID]:
    # relative
    from .action_object import ActionObject
    from .twin_object import TwinObject

    uid_kwargs = {}
    for k, v in kwargs.items():
        uid = v
        if isinstance(v, ActionObject):
            uid = v.id
        if isinstance(v, TwinObject):
            uid = v.id
        if isinstance(v, Asset):
            uid = v.action_id

        if not isinstance(uid, UID):
            raise Exception(f"Input {k} must have a UID not {type(v)}")

        uid_kwargs[k] = uid
    return uid_kwargs


class InputPolicy(SyftObject):
    # version
    __canonical_name__ = "InputPolicy"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    inputs: Dict[str, Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        uid = UID()
        if "id" in kwargs:
            uid = kwargs["id"]
        if "inputs" in kwargs:
            kwargs = kwargs["inputs"]
        uid_kwargs = extract_uids(kwargs)
        super().__init__(id=uid, inputs=uid_kwargs)

    def filter_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


def allowed_ids_only(
    allowed_inputs: Dict[str, UID], kwargs: Dict[str, Any]
) -> Dict[str, UID]:
    filtered_kwargs = {}
    for key in allowed_inputs.keys():
        if key in kwargs:
            value = kwargs[key]
            uid = value
            if not isinstance(uid, UID):
                uid = getattr(value, "id", None)

            if uid != allowed_inputs[key]:
                raise Exception(
                    f"Input {type(value)} for {key} not in allowed {allowed_inputs}"
                )
            filtered_kwargs[key] = value
    return filtered_kwargs


@serializable(recursive_serde=True)
class ExactMatch(InputPolicy):
    # version
    __canonical_name__ = "ExactMatch"
    __version__ = SYFT_OBJECT_VERSION_1

    def filter_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return allowed_ids_only(self.inputs, kwargs)


class OutputPolicyState(SyftObject):
    # version
    __canonical_name__ = "OutputPolicyState"
    __version__ = SYFT_OBJECT_VERSION_1

    @property
    def valid(self) -> Union[SyftSuccess, SyftError]:
        raise NotImplementedError

    def update_state(self) -> None:
        raise NotImplementedError


@serializable(recursive_serde=True)
class OutputPolicyStateExecuteCount(OutputPolicyState):
    # version
    __canonical_name__ = "OutputPolicyStateExecuteCount"
    __version__ = SYFT_OBJECT_VERSION_1

    count: int = 0
    limit: int

    @property
    def valid(self) -> Union[SyftSuccess, SyftError]:
        is_valid = self.count < self.limit
        if is_valid:
            return SyftSuccess(
                message=f"Policy is still valid. count: {self.count} < limit: {self.limit}"
            )
        return SyftError(
            message=f"Policy is no longer valid. count: {self.count} >= limit: {self.limit}"
        )

    def update_state(self) -> None:
        if self.count >= self.limit:
            raise Exception(
                f"Update state being called with count: {self.count} "
                f"beyond execution limit: {self.limit}"
            )
        self.count += 1


@serializable(recursive_serde=True)
class OutputPolicyStateExecuteOnce(OutputPolicyStateExecuteCount):
    __canonical_name__ = "OutputPolicyStateExecuteOnce"
    __version__ = SYFT_OBJECT_VERSION_1

    limit: int = 1


class OutputPolicy(SyftObject):
    # version
    __canonical_name__ = "OutputPolicy"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    state_type: Optional[Type[OutputPolicyState]]

    def update() -> None:
        raise NotImplementedError


@serializable(recursive_serde=True)
class SingleExecutionExactOutput(OutputPolicy):
    # version
    __canonical_name__ = "SingleExecutionExactOutput"
    __version__ = SYFT_OBJECT_VERSION_1

    state_type: Type[OutputPolicyState] = OutputPolicyStateExecuteOnce


@serializable(recursive_serde=True)
class UserPolicyStatus(Enum):
    SUBMITTED = "submitted"
    DENIED = "denied"
    APPROVED = "approved"

@serializable(recursive_serde=True)
class UserPolicy(SyftObject):
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
    status: UserPolicyStatus = UserPolicyStatus.SUBMITTED
    
    
@serializable(recursive_serde=True)
class SubmitUserPolicy(SyftObject):
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

# @transform(CreatePolicy, Policy)
# def create_policy_to_policy() -> List[Callable]:
#     return [
#         generate_id,
#         add_credentials_for_key("user_verify_key")
#         # check_policy_uid,
#         # check_user_uid,
#         # create_object
#     ]

@transform(SubmitUserPolicy, UserPolicy)
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
    
def init_policy(user_policy: UserPolicy, init_args: Dict[str, Any]):
    exec(user_policy.raw_code)
    policy_class_name = eval(user_policy.class_name)
    policy_object = policy_class_name(**init_args)
    return policy_object 
    
def get_policy_object(user_policy: UserPolicy, state: str) -> Result[Any,str]:
    # if user_policy.status != UserPolicyStatus.APPROVED:
    #     return Err() 
    exec(user_policy.raw_code)
    policy_class_name = eval(user_policy.class_name)
    policy_object = _deserialize(state, from_bytes=True, class_type=policy_class_name)
    return policy_object#Ok(policy_object)

def update_policy_state(policy_object):
    return _serialize(policy_object, to_bytes=True)
