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
from typing import Type
from typing import Union

# relative
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .context import NodeServiceContext
from .credentials import SyftVerifyKey
from .dataset import Asset
from .datetime import DateTime
from .document_store import PartitionKey
from .response import SyftError
from .response import SyftSuccess
from .transforms import TransformContext
from .transforms import generate_id
from .transforms import transform
from .user_code_parse import GlobalsVisitor

UserVerifyKeyPartitionKey = PartitionKey(key="user_verify_key", type_=SyftVerifyKey)
CodeHashPartitionKey = PartitionKey(key="code_hash", type_=int)

PyCodeObject = Any


class InputPolicy(SyftObject):
    # version
    __canonical_name__ = "InputPolicy"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    inputs: Dict[str, Any]
    node_uid: Optional[UID]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        uid = UID()
        node_uid = None
        if "id" in kwargs:
            uid = kwargs["id"]
        if "node_uid" in kwargs:
            node_uid = kwargs["node_uid"]

        # finally get inputs
        if "inputs" in kwargs:
            kwargs = kwargs["inputs"]
        uid_kwargs = extract_uids(kwargs)

        super().__init__(id=uid, inputs=uid_kwargs, node_uid=node_uid)

    def filter_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def __getitem__(self, key: Union[int, str]) -> Optional[SyftObject]:
        if isinstance(key, int):
            key = list(self.inputs.keys())[key]
        uid = self.inputs[key]
        # TODO Add NODE UID or LINK so we can resolve this object
        return uid

    @property
    def assets(self) -> List[Asset]:
        # relative
        from .api import APIRegistry

        api = APIRegistry.api_for(self.node_uid)
        if api is None:
            return SyftError(message=f"You must login to {self.node_uid}")

        all_assets = []
        for k, uid in self.inputs.items():
            if isinstance(uid, UID):
                assets = api.services.dataset.get_assets_by_action_id(uid)
                if not isinstance(assets, list):
                    return assets

                all_assets += assets
        return all_assets


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


@serializable(recursive_serde=True)
class OutputHistory(SyftObject):
    # version
    __canonical_name__ = "OutputHistory"
    __version__ = SYFT_OBJECT_VERSION_1

    output_time: DateTime
    outputs: Optional[Union[List[UID], Dict[str, UID]]]
    executing_user_verify_key: SyftVerifyKey


class OutputPolicyState(SyftObject):
    # version
    __canonical_name__ = "OutputPolicyState"
    __version__ = SYFT_OBJECT_VERSION_1

    output_history: List[OutputHistory] = []

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

    def update_state(
        self,
        context: NodeServiceContext,
        outputs: Optional[Union[UID, List[UID], Dict[str, UID]]],
    ) -> None:
        if self.count >= self.limit:
            raise Exception(
                f"Update state being called with count: {self.count} "
                f"beyond execution limit: {self.limit}"
            )
        if isinstance(outputs, UID):
            outputs = [outputs]
        history = OutputHistory(
            output_time=DateTime.now(),
            outputs=outputs,
            executing_user_verify_key=context.credentials,
        )
        self.output_history.append(history)
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
    outputs: List[str] = []
    state_type: Optional[Type[OutputPolicyState]]

    def update() -> None:
        raise NotImplementedError

    @property
    def policy_code(self) -> str:
        cls = type(self)
        op_code = inspect.getsource(cls)
        if self.state_type:
            state_code = inspect.getsource(self.state_type)
            op_code += "\n" + state_code
        return op_code


@serializable(recursive_serde=True)
class SingleExecutionExactOutput(OutputPolicy):
    # version
    __canonical_name__ = "SingleExecutionExactOutput"
    __version__ = SYFT_OBJECT_VERSION_1

    state_type: Type[OutputPolicyState] = OutputPolicyStateExecuteOnce


@serializable(recursive_serde=True)
class UserCodeStatus(Enum):
    SUBMITTED = "submitted"
    DENIED = "denied"
    EXECUTE = "execute"


@serializable(recursive_serde=True)
class UserCode(SyftObject):
    # version
    __canonical_name__ = "UserCode"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    user_verify_key: SyftVerifyKey
    raw_code: str
    input_policy: InputPolicy
    output_policy: OutputPolicy
    output_policy_state: OutputPolicyState
    parsed_code: str
    service_func_name: str
    unique_func_name: str
    user_unique_func_name: str
    code_hash: str
    signature: inspect.Signature
    status: UserCodeStatus = UserCodeStatus.SUBMITTED

    __attr_searchable__ = ["user_verify_key", "status", "service_func_name"]
    __attr_unique__ = ["code_hash", "user_unique_func_name"]
    __attr_repr_cols__ = ["status", "service_func_name"]

    @property
    def byte_code(self) -> Optional[PyCodeObject]:
        return compile_byte_code(self.parsed_code)

    @property
    def unsafe_function(self) -> Optional[Callable]:
        print("WARNING: This code was submitted by a User and could be UNSAFE.")

        # 🟡 TODO: re-use the same infrastructure as the execute_byte_code function
        def wrapper(*args: Any, **kwargs: Any) -> Callable:
            # remove the decorator
            inner_function = ast.parse(self.raw_code).body[0]
            inner_function.decorator_list = []
            # compile the function
            raw_byte_code = compile_byte_code(ast.unparse(inner_function))
            # load it
            exec(raw_byte_code)  # nosec
            # execute it
            evil_string = f"{self.service_func_name}(*args, **kwargs)"
            result = eval(evil_string, None, locals())  # nosec
            # return the results
            return result

        return wrapper

    @property
    def code(self) -> str:
        return self.raw_code


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


@serializable(recursive_serde=True)
class SubmitUserCode(SyftObject):
    # version
    __canonical_name__ = "SubmitUserCode"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    code: str
    func_name: str
    signature: inspect.Signature
    input_policy: InputPolicy
    output_policy: OutputPolicy
    local_function: Optional[Callable]

    __attr_state__ = [
        "id",
        "code",
        "func_name",
        "signature",
        "input_policy",
        "output_policy",
    ]

    @property
    def kwargs(self) -> List[str]:
        return self.input_policy.inputs

    @property
    def outputs(self) -> List[str]:
        return self.output_policy.outputs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # only run this on the client side
        if self.local_function:
            # filtered_args = []
            filtered_kwargs = {}
            # for arg in args:
            #     filtered_args.append(debox_asset(arg))
            for k, v in kwargs.items():
                filtered_kwargs[k] = debox_asset(v)

            return self.local_function(**filtered_kwargs)
        else:
            raise NotImplementedError


def debox_asset(arg: Any) -> Any:
    deboxed_arg = arg
    if isinstance(deboxed_arg, Asset):
        deboxed_arg = arg.mock
    if hasattr(deboxed_arg, "syft_action_data"):
        deboxed_arg = deboxed_arg.syft_action_data
    return deboxed_arg


def syft_function(input_policy, output_policy) -> SubmitUserCode:
    def decorator(f):
        return SubmitUserCode(
            code=inspect.getsource(f),
            func_name=f.__name__,
            signature=inspect.signature(f),
            input_policy=input_policy,
            output_policy=output_policy,
            local_function=f,
        )

    return decorator


def generate_unique_func_name(context: TransformContext) -> TransformContext:
    code_hash = context.output["code_hash"]
    service_func_name = context.output["func_name"]
    context.output["service_func_name"] = service_func_name
    func_name = f"user_func_{service_func_name}_{context.credentials}_{code_hash}"
    user_unique_func_name = f"user_func_{service_func_name}_{context.credentials}"
    context.output["unique_func_name"] = func_name
    context.output["user_unique_func_name"] = user_unique_func_name
    return context


def process_code(
    raw_code: str,
    func_name: str,
    original_func_name: str,
    input_policy: InputPolicy,
    output_policy: OutputPolicy,
) -> str:
    input_kwargs = input_policy.inputs
    outputs = output_policy.outputs

    tree = ast.parse(raw_code)

    # check there are no globals
    v = GlobalsVisitor()
    v.visit(tree)

    f = tree.body[0]
    f.decorator_list = []

    keywords = [ast.keyword(arg=i, value=[ast.Name(id=i)]) for i in input_kwargs]
    call_stmt = ast.Assign(
        targets=[ast.Name(id="result")],
        value=ast.Call(
            func=ast.Name(id=original_func_name), args=[], keywords=keywords
        ),
        lineno=0,
    )

    if len(outputs) > 0:
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
        # requires typing module imported but main code returned is FunctionDef not Module
        # return_annotation = ast.parse("typing.Dict[str, typing.Any]", mode="eval").body
    else:
        return_stmt = ast.Return(value=ast.Name(id="result"))
        # requires typing module imported but main code returned is FunctionDef not Module
        # return_annotation = ast.parse("typing.Any", mode="eval").body

    new_body = tree.body + [call_stmt, return_stmt]

    wrapper_function = ast.FunctionDef(
        name=func_name,
        args=f.args,
        body=new_body,
        decorator_list=[],
        returns=None,
        lineno=0,
    )

    return ast.unparse(wrapper_function)


def new_check_code(context: TransformContext) -> TransformContext:
    try:
        processed_code = process_code(
            raw_code=context.output["raw_code"],
            func_name=context.output["unique_func_name"],
            original_func_name=context.output["service_func_name"],
            input_policy=context.output["input_policy"],
            output_policy=context.output["output_policy"],
        )
        context.output["parsed_code"] = processed_code

    except Exception as e:
        raise e

    return context


def compile_byte_code(parsed_code: str) -> Optional[PyCodeObject]:
    try:
        return compile(parsed_code, "<string>", "exec")
    except Exception as e:
        print("WARNING: to compile byte code", e)
    return None


def compile_code(context: TransformContext) -> TransformContext:
    byte_code = compile_byte_code(context.output["parsed_code"])
    if byte_code is None:
        raise Exception(
            "Unable to compile byte code from parsed code. "
            + context.output["parsed_code"]
        )
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
        for k in context.output["input_policy"].inputs.keys()
    ]
    sig = Signature(parameters=params)
    context.output["signature"] = sig
    return context


def modify_signature(context: TransformContext) -> TransformContext:
    sig = context.output["signature"]
    context.output["signature"] = sig.replace(return_annotation=Dict[str, Any])
    return context


def check_input_policy(context: TransformContext) -> TransformContext:
    ip = context.output["input_policy"]
    ip.node_uid = context.node.id
    context.output["input_policy"] = ip
    return context


def init_output_policy_state(context: TransformContext) -> TransformContext:
    context.output["output_policy_state"] = context.output["output_policy"].state_type()
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
        check_input_policy,
        init_output_policy_state,
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
        print("execute_byte_code failed", e, file=stderr_)
    finally:
        sys.stdout = stdout_
        sys.stderr = stderr_
