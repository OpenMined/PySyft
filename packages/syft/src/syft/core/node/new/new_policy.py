from typing import Dict
from typing import Any
from typing import Optional
from enum import Enum
from typing import Union
from typing import Type
from RestrictedPython import compile_restricted
from result import Ok
from result import Result
from typing import List
import sys
from inspect import Parameter
from inspect import Signature
import hashlib
from typing import Callable
import ast
from io import StringIO
import sys

from .credentials import SyftVerifyKey
from .syft_object import SYFT_OBJECT_VERSION_1
from .syft_object import SyftObject
import inspect
from .serializable import serializable
from .transforms import generate_id
from .transforms import transform
from .uid import UID
from .policy_code_parse import GlobalsVisitor
from .response import SyftError
from .response import SyftSuccess
from .datetime import DateTime
from .context import NodeServiceContext
from .context import AuthedServiceContext
from .policy import allowed_ids_only, retrieve_from_db
from .api import NodeView
from .transforms import TransformContext
from .deserialize import _deserialize
from .serialize import _serialize

PyCodeObject = Any

@serializable(recursive_serde=True)
class OutputHistory(SyftObject):
    # version
    __canonical_name__ = "OutputHistory_2"
    __version__ = SYFT_OBJECT_VERSION_1

    output_time: DateTime
    outputs: Optional[Union[List[UID], Dict[str, UID]]]
    executing_user_verify_key: SyftVerifyKey

@serializable(recursive_serde=True)
class UserPolicyStatus(Enum):
    SUBMITTED = "submitted"
    DENIED = "denied"
    APPROVED = "approved"

class Policy:
    id: UID
    @property
    def policy_code(self) -> str:
        cls = type(self)
        op_code = inspect.getsource(cls)
        return op_code
    
    def public_state() -> None:
        raise NotImplementedError
    
class InputPolicy(Policy):
    inputs: Dict[NodeView, Any]
    # node_uid: Optional[UID]
    
    def filter_kwargs() -> None:
        raise NotImplementedError


@serializable(recursive_serde=True)
class ExactMatch(InputPolicy, SyftObject):
    # version
    __canonical_name__ = "ExactMatch_2"
    __version__ = SYFT_OBJECT_VERSION_1

    def filter_kwargs(
        self, kwargs: Dict[str, Any], context: AuthedServiceContext, code_item_id: UID
    ) -> Dict[str, Any]:
        allowed_inputs = allowed_ids_only(
            allowed_inputs=self.inputs, kwargs=kwargs, context=context
        )
        return retrieve_from_db(
            code_item_id=code_item_id, allowed_inputs=allowed_inputs, context=context
        )

class OutputPolicy(Policy):
    
    output_history: List[OutputHistory] = []
    outputs: List[str] = []
    node_uid: Optional[UID]

    def apply_output(
        self, 
        context: NodeServiceContext,
        outputs: Union[UID, List[UID], Dict[str, UID]],
    ) -> None:
        if isinstance(outputs, UID):
            outputs = [outputs]
        history = OutputHistory(
            output_time=DateTime.now(),
            outputs=outputs,
            executing_user_verify_key=context.credentials,
        )
        self.output_history.append(history)

@serializable(recursive_serde=True)
class OutputPolicyExecuteCount(OutputPolicy, SyftObject):
    __canonical_name__ = "OutputPolicyExecuteCount_2"
    __version__ = SYFT_OBJECT_VERSION_1

    count: int = 0
    limit: int
    
    def apply_output(
        self, 
        context: NodeServiceContext, 
        outputs: Union[UID, List[UID], Dict[str, UID]]
    ) -> Union[UID, List[UID], Dict[str, UID]]:
        if self.count < self.limit:
            super().apply_output(context, outputs)
            self.count +=1
            return SyftSuccess()
        else:
            return  SyftError(
                message=f"Policy is no longer valid. count: {self.count} >= limit: {self.limit}"
            )
            
    def public_state(self) -> None:
        return {"limit": self.limit, "count": self.count}
            
@serializable(recursive_serde=True)
class OutputPolicyExecuteOnce(OutputPolicyExecuteCount):
    __canonical_name__ = "OutputPolicyExecuteOnce_2"
    __version__ = SYFT_OBJECT_VERSION_1

    limit: int = 1


class CustomPolicy(Policy):
    init_args: Dict[str, Any] = {}
    init_kwargs: Dict[str, Any] = {}

    def __init__(self, *args, **kwargs) -> None:
        self.init_args = args
        self.init_kwargs = kwargs

    
class CustomInputPolicy(CustomPolicy, InputPolicy):
    pass
    
class CustomOutputPolicy(CustomPolicy, OutputPolicy):
    pass


 
@serializable(recursive_serde=True)
class UserPolicy(SyftObject, Policy):
    __canonical_name__ = "UserPolicy_2"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    node_uid: Optional[UID]
    user_verify_key: SyftVerifyKey
    raw_code: str
    parsed_code: str
    signature: inspect.Signature
    class_name: str
    unique_name: str
    code_hash: str
    byte_code: PyCodeObject
    status: UserPolicyStatus = UserPolicyStatus.SUBMITTED
    state_type: Optional[Type] = None

    @property
    def byte_code(self) -> Optional[PyCodeObject]:
        return compile_byte_code(self.parsed_code)

    @property
    def valid(self) -> Union[SyftSuccess, SyftError]:
        return SyftSuccess(message="Policy is valid.")
    
    @property
    def policy_code(self) -> str:
        return self.raw_code

def compile_byte_code(parsed_code: str) -> Optional[PyCodeObject]:
    try:
        return compile(parsed_code, "<string>", "exec")
    except Exception as e:
        print("WARNING: to compile byte code", e)
    return None

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


def process_class_code(
    raw_code: str, 
    class_name: str,
    input_kwargs: List[str]
) -> str:
    print("process_class_code", file=sys.stderr)
    tree = ast.parse(raw_code)

    v = GlobalsVisitor()
    v.visit(tree)

    print(ast.dump(ast.parse(raw_code), indent=4), file=sys.stderr)
    # print(tree, file=sys.stderr)
    # print(tree.body[0], file=sys.stderr)
    # print(tree.body[0].module, file=sys.stderr)
    # print(tree.body[0].names[0].asname, file=sys.stderr)
    # print(tree.body[0].level, file=sys.stderr)
    # print(tree.body[1], file=sys.stderr)
    # print(tree.body[1].name.id, tree.body[1].name.ctx, file=sys.stderr)
    print(tree.body[1].bases, file=sys.stderr)
    # print(tree.body[1].keywords, file=sys.stderr)
    # print(tree.body[1].body, file=sys.stderr)
    # print(tree.body[1].decorator_list[0], file=sys.stderr)
    serializable_name = ast.Name(id='serializable', ctx=ast.Load())
    serializable_decorator = ast.Call(
                                func=serializable_name, 
                                args=[], 
                                keywords=[ast.keyword(
                                    arg='recursive_serde',
                                    value=ast.Constant(value=True)    
                                )]
                            )
    print(ast.dump(serializable_decorator, indent=4), file=sys.stderr)
    # print(serializable_decorator == tree.body[1].decorator_list[0], file=sys.__stderr__)

    f = tree.body[1]
    f.decorator_list = [serializable_decorator]
    print(ast.unparse(f), file=sys.stderr)
    print(raw_code, file=sys.stderr)
    return raw_code


def check_class_code(context: TransformContext) -> TransformContext:
    # TODO: define the proper checking for this case based on the ideas from UserCode
    # check for no globals
    # check for Policy template -> __init__, apply_output, public_state
    # parse init signature
    # check dangerous libraries, maybe compile_restricted already does that
    try:
        print("check_class_code", file=sys.stderr) 
        # print(context.output, file=sys.stderr)
        processed_code = process_class_code(
            raw_code=context.output['raw_code'],
            class_name=context.output['unique_name'],
            input_kwargs=context.output["input_kwargs"],
        )
        context.output["parsed_code"] = processed_code
    
    except Exception as e:
        raise e
    return context


def compile_code(context: TransformContext) -> TransformContext:
    print("compile_code", file=sys.stderr) 
    byte_code = compile_byte_code(context.output["parsed_code"])
    if byte_code is None:
        raise Exception(
            "Unable to compile byte code from parsed code. "
            + context.output["parsed_code"]
        )
    return context


def add_credentials_for_key(key: str) -> Callable:
    
    print("add_credentials_for_key", file=sys.stderr) 
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


def serialization_addon(context: TransformContext) -> TransformContext:
    print(context.output['code'], file=sys.stderr)

    policy_code = context.output['code']
    context.output['code']= '@serializable(recursive_serde=True)\n' + policy_code 
    return context

@serializable(recursive_serde=True)
class SubmitUserPolicy(SyftObject, Policy):
    __canonical_name__ = "SubmitUserPolicy_2"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    code: str
    class_name: str
    input_kwargs: List[str]

    def compile(self) -> PyCodeObject:
        return compile_restricted(self.code, "<string>", "exec")

@transform(SubmitUserPolicy, UserPolicy)
def submit_policy_code_to_user_code() -> List[Callable]:
    return [
        generate_id,
        # serialization_addon,
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
        # syft absolute
        import syft as sy  # noqa: F401 # provide sy.Things to user code

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
    policy_object = policy_class(**init_args)
    return policy_object


def get_policy_object(user_policy: UserPolicy, state: str) -> Result[Any, str]:
    policy_class = execute_policy_code(user_policy)
    policy_object = _deserialize(state, from_bytes=True, class_type=policy_class)
    return policy_object


def update_policy_state(policy_object):
    return _serialize(policy_object, to_bytes=True)

