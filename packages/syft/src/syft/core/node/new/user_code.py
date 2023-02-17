# stdlib
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

# relative
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .credentials import SyftVerifyKey
from .document_store import PartitionKey
from .transforms import TransformContext
from .transforms import generate_id
from .transforms import transform
from .user_code_parse import parse_and_wrap_code

UserVerifyKeyPartitionKey = PartitionKey(key="user_verify_key", type_=SyftVerifyKey)
CodeHashPartitionKey = PartitionKey(key="code_hash", type_=int)

stdout_ = sys.stdout
stderr_ = sys.stderr

PyCodeObject = Any


@serializable(recursive_serde=True)
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
    raw_code: str
    input_kwargs: List[str]
    output_args: List[str]
    parsed_code: str
    service_func_name: str
    unique_func_name: str
    code_hash: str
    signature: inspect.Signature
    byte_code: PyCodeObject
    status: UserCodeStatus = UserCodeStatus.SUBMITTED

    __attr_searchable__ = ["status", "service_func_name"]
    __attr_unique__ = ["user_verify_key", "code_hash", "unique_func_name"]


@serializable(recursive_serde=True)
class SubmitUserCode(SyftObject):
    # version
    __canonical_name__ = "SubmitUserCode"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    code: str
    func_name: str
    input_kwargs: List[str]
    output_args: List[str]

    def compile(self) -> PyCodeObject:
        return compile_restricted(self.code, "<string>", "exec")


def generate_unique_func_name(context: TransformContext) -> TransformContext:
    code_hash = context.output["code_hash"]
    service_func_name = context.output["func_name"]
    context.output["service_func_name"] = service_func_name
    func_name = f"user_func_{service_func_name}_{context.credentials}_{code_hash}"
    context.output["unique_func_name"] = func_name
    return context


def check_code(context: TransformContext) -> TransformContext:
    parsed_code = parse_and_wrap_code(
        func_name=context.output["unique_func_name"],
        raw_code=context.output["raw_code"],
        input_kwargs=context.output["input_kwargs"],
        output_args=context.output["output_args"],
    )
    context.output["parsed_code"] = parsed_code
    return context


def compile_code(context: TransformContext) -> TransformContext:
    # byte_code = compile_restricted(context.output["parsed_code"], "<string>", "exec")
    byte_code = compile(context.output["parsed_code"], "<string>", "exec")
    context.output["byte_code"] = byte_code
    return context


def hash_code(context: TransformContext) -> TransformContext:
    code = context.output["code"]
    del context.output["code"]
    context.output["raw_code"] = code
    code_hash = hashlib.sha256(code.encode("utf8")).hexdigest()
    context.output["code_hash"] = code_hash
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


@transform(SubmitUserCode, UserCode)
def submit_user_code_to_user_code() -> List[Callable]:
    return [
        generate_id,
        hash_code,
        generate_unique_func_name,
        generate_signature,
        check_code,
        compile_code,
        add_credentials_for_key("user_verify_key"),
    ]


@serializable(recursive_serde=True)
class UserCodeExecutionResult(SyftObject):
    # version
    __canonical_name__ = "UserCodeExecutionResult"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    user_code_id: UID
    stdout: str
    stderr: str
    result: Any


def execute_byte_code(code_item: UserCode, kwargs: Dict[str, Any]) -> Any:
    global stdout_
    global stderr_
    try:
        stdout = StringIO()
        stderr = StringIO()

        sys.stdout = stdout
        sys.stderr = stderr

        # statisfy lint checker
        result = None

        exec(code_item.byte_code)  # nosec
        evil_string = f"result = {code_item.unique_func_name}(**kwargs)"

        # copy locals
        _locals = locals()

        # pass in kwargs and evaluate
        exec(evil_string, None, _locals)  # nosec

        # assign result back to local scope
        result = _locals["result"]

        # restore stdout and stderr
        sys.stdout = stdout_
        sys.stderr = stderr_

        return UserCodeExecutionResult(
            user_code_id=code_item.id,
            stdout=str(stdout.getvalue()),
            stderr=str(stderr.getvalue()),
            result=result,
        )

    except Exception as e:
        sys.stdout = stdout_
        sys.stderr = stderr_
        print("execute_byte_code failed", e)
    finally:
        sys.stdout = stdout_
        sys.stderr = stderr_
