# stdlib
import ast
from enum import Enum
import hashlib
import inspect
from inspect import Parameter
from inspect import Signature
from io import StringIO
import sys
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

# third party
from RestrictedPython import compile_restricted
from result import Result

# relative
from .credentials import SyftVerifyKey
from .deserialize import _deserialize
from .document_store import PartitionKey
from .policy_code_parse import GlobalsVisitor
from .serializable import serializable
from .serialize import _serialize
from .syft_object import SYFT_OBJECT_VERSION_1
from .syft_object import SyftObject
from .transforms import TransformContext
from .transforms import generate_id
from .transforms import transform
from .uid import UID

PolicyUserVerifyKeyPartitionKey = PartitionKey(
    key="user_verify_key", type_=SyftVerifyKey
)
PyCodeObject = Any


@serializable(recursive_serde=True)
class UserPolicyStatus(Enum):
    SUBMITTED = "submitted"
    DENIED = "denied"
    APPROVED = "approved"


@serializable(recursive_serde=True)
class UserPolicy(SyftObject):
    __canonical_name__ = "UserPolicy"
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

    @property
    def byte_code(self) -> Optional[PyCodeObject]:
        return compile_byte_code(self.parsed_code)


@serializable(recursive_serde=True)
class SubmitUserPolicy(SyftObject):
    __canonical_name__ = "SubmitUserPolicy"
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


def compile_byte_code(parsed_code: str) -> Optional[PyCodeObject]:
    try:
        return compile(parsed_code, "<string>", "exec")
    except Exception as e:
        print("WARNING: to compile byte code", e)
    return None


def process_class_code(raw_code: str, class_name: str) -> str:
    tree = ast.parse(raw_code)

    v = GlobalsVisitor()
    v.visit(tree)

    f = tree.body[0]
    f.decorator_list = []


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
    byte_code = compile_byte_code(context.output["parsed_code"])
    if byte_code is None:
        raise Exception(
            "Unable to compile byte code from parsed code. "
            + context.output["parsed_code"]
        )
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


@transform(SubmitUserPolicy, UserPolicy)
def submit_policy_code_to_user_code() -> List[Callable]:
    return [
        generate_id,
        hash_code,
        generate_unique_class_name,
        generate_signature,
        check_class_code,
        compile_code,
        add_credentials_for_key("user_verify_key"),
    ]


def execute_policy_code(user_policy: UserPolicy):
    # print(user_policy.raw_code, file=sys.stderr)
    stdout_ = sys.stdout
    stderr_ = sys.stderr

    try:
        stdout = StringIO()
        stderr = StringIO()

        sys.stdout = stdout
        sys.stderr = stderr
        exec(user_policy.byte_code)  # nosec
        policy_class = eval(user_policy.class_name)  # nosec

        sys.stdout = stdout_
        sys.stderr = stderr_

        return policy_class

    except Exception as e:
        print("execute_byte_code failed", e, file=stderr_)
        try:
            stdout = StringIO()
            stderr = StringIO()

            sys.stdout = stdout
            sys.stderr = stderr
            # exec(user_policy.byte_code)  # nosec
            # policy_class = eval(user_policy.class_name)  # nosec
            print(
                user_policy.__object_version_registry__["RepeatedCallPolicy_1"],
                file=stderr_,
            )
            policy_class = user_policy.__object_version_registry__[
                "RepeatedCallPolicy_1"
            ]

            sys.stdout = stdout_
            sys.stderr = stderr_

            return policy_class
        except Exception as e:
            print("execute_byte_code failed", e, file=stderr_)

    finally:
        sys.stdout = stdout_
        sys.stderr = stderr_


def init_policy(user_policy: UserPolicy, init_args: Dict[str, Any]):
    policy_class = execute_policy_code(user_policy)
    print(policy_class, file=sys.stderr)
    print(init_args, file=sys.stderr)
    policy_object = policy_class(**init_args)
    return policy_object


def get_policy_object(user_policy: UserPolicy, state: str) -> Result[Any, str]:
    policy_class = execute_policy_code(user_policy)
    policy_object = _deserialize(state, from_bytes=True, class_type=policy_class)
    return policy_object


def update_policy_state(policy_object):
    return _serialize(policy_object, to_bytes=True)
