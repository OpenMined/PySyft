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
    output_arg: str
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
class ExactMatch(SyftObject):
    # version
    __canonical_name__ = "ExactMatch"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    inputs: Dict[str, Any]

    # def __init__(self, *args: Any, **kwargs: Any) -> None:
    #     super().__init__(inputs=kwargs)


@serializable(recursive_serde=True)
class SingleExecutionExactOutput(SyftObject):
    # version
    __canonical_name__ = "SingleExecutionExactOutput"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    outputs: Optional[List[str]]


@serializable(recursive_serde=True)
class SubmitUserCode(SyftObject):
    # version
    __canonical_name__ = "SubmitUserCode"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    code: str
    func_name: str
    signature: inspect.Signature
    input_policy: ExactMatch
    output_policy: SingleExecutionExactOutput

    @property
    def kwargs(self) -> List[str]:
        return self.input_policy.inputs

    @property
    def outputs(self) -> List[str]:
        return self.output_policy.outputs


def syft_function(input_policy, output_policy) -> SubmitUserCode:
    def decorator(f):
        return SubmitUserCode(
            code=inspect.getsource(f),
            func_name=f.__name__,
            signature=inspect.signature(f),
            input_policy=input_policy,
            output_policy=output_policy,
        )

    return decorator


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
        output_arg=context.output["output_arg"],
    )
    context.output["parsed_code"] = parsed_code
    return context


def new_check_code(context: TransformContext) -> TransformContext:
    raw_code = context.output["raw_code"]
    func_name = context.output["unique_func_name"]
    service_func_name = context.output["service_func_name"]

    inputs = context.output["input_policy"].inputs
    input_kwargs = list(inputs.keys())

    outputs = context.output["output_policy"].outputs

    tree = ast.parse(raw_code)
    f = tree.body[0]
    f.decorator_list = []

    keywords = [ast.keyword(arg=i, value=[ast.Name(id=i)]) for i in input_kwargs]
    call_stmt = ast.Assign(
        targets=[ast.Name(id="result")],
        value=ast.Call(func=ast.Name(id=service_func_name), args=[], keywords=keywords),
        lineno=0,
    )

    output_list = ast.List(elts=[ast.Constant(value=x) for x in outputs])
    return_stmt = ast.Return(
        value=ast.DictComp(
            key=ast.Name(id="k"),
            value=ast.Subscript(
                value=ast.Name(id="result"),
                slice=ast.Name(id="k"),
            ),
            generators=[
                ast.comprehension(
                    target=ast.Name(id="k"), iter=output_list, ifs=[], is_async=0
                )
            ],
        )
    )

    new_body = tree.body + [call_stmt, return_stmt]

    return_annotation = ast.parse("Dict[str, Any]", mode="eval").body

    wrapper_function = ast.FunctionDef(
        name=func_name,
        args=f.args,
        body=new_body,
        decorator_list=[],
        returns=return_annotation,
        lineno=0,
    )

    context.output["parsed_code"] = ast.unparse(wrapper_function)
    context.output["input_kwargs"] = input_kwargs
    context.output["output_arg"] = outputs[0]

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
    params = [
        Parameter(name=k, kind=Parameter.POSITIONAL_OR_KEYWORD)
        for k in context.output["input_kwargs"]
    ]
    sig = Signature(parameters=params)
    context.output["signature"] = sig
    return context


def modify_signature(context: TransformContext) -> TransformContext:
    sig = context.output["signature"]
    context.output["signature"] = sig.replace(return_annotation=Dict[str, Any])
    return context


@transform(SubmitUserCode, UserCode)
def submit_user_code_to_user_code() -> List[Callable]:
    return [
        generate_id,
        hash_code,
        generate_unique_func_name,
        modify_signature,
        new_check_code,
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
    stdout_ = sys.stdout
    stderr_ = sys.stderr

    try:
        stdout = StringIO()
        stderr = StringIO()

        sys.stdout = stdout
        sys.stderr = stderr

        # statisfy lint checker
        result = None

        exec(code_item.byte_code)  # nosec

        evil_string = f"{code_item.unique_func_name}(**kwargs)"
        result = eval(evil_string, None, locals())  # nosec

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
