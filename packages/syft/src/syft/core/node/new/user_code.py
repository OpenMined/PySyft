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
from typing import Union

# relative
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .credentials import SyftVerifyKey
from .dataset import Asset
from .document_store import PartitionKey
from .response import SyftError
from .response import SyftSuccess
from .transforms import TransformContext
from .transforms import generate_id
from .transforms import transform
from .user_code_parse import GlobalsVisitor
from .policy import UserPolicy, SubmitUserPolicy, InputPolicy, OutputPolicy, OutputPolicyState
from IPython.core.magics.code import extract_symbols
from ....util import is_interpreter_jupyter

UserVerifyKeyPartitionKey = PartitionKey(key="user_verify_key", type_=SyftVerifyKey)
CodeHashPartitionKey = PartitionKey(key="code_hash", type_=int)

PyCodeObject = Any


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
    input_policy: Union[InputPolicy, UserPolicy]
    input_policy_state: Union[OutputPolicyState, str]
    output_policy: Union[OutputPolicy, UserPolicy]
    output_policy_state: Union[OutputPolicyState, str]
    parsed_code: str
    service_func_name: str
    unique_func_name: str
    user_unique_func_name: str
    code_hash: str
    signature: inspect.Signature
    status: UserCodeStatus = UserCodeStatus.SUBMITTED
    input_kwargs: List[str]
    outputs: List[str]

    __attr_searchable__ = ["status", "service_func_name"]
    __attr_unique__ = ["user_verify_key", "code_hash", "user_unique_func_name"]
    __attr_repr_cols__ = ["status", "service_func_name"]

    @property
    def byte_code(self) -> Optional[PyCodeObject]:
        return compile_byte_code(self.parsed_code)

    @property
    def code(self) -> str:
        return self.raw_code


@serializable(recursive_serde=True)
class SubmitUserCode(SyftObject):
    # version
    __canonical_name__ = "SubmitUserCode"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    code: str
    func_name: str
    signature: inspect.Signature
    input_policy: Union[InputPolicy, SubmitUserPolicy]
    input_policy_init_args: Dict[str, Any]
    output_policy: Union[OutputPolicy, SubmitUserPolicy]
    output_policy_init_args: Dict[str, Any]
    local_function: Optional[Callable]
    input_kwargs: List[str]
    outputs: List[str]
    
    __attr_state__ = [
        "id",
        "code",
        "func_name",
        "signature",
        "input_policy",
        "output_policy",
    ]

    # @property
    # def kwargs(self) -> List[str]:
    #     return self.input_policy.inputs

    # @property
    # def outputs(self) -> List[str]:
    #     return self.output_policy.outputs

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


def new_getfile(object):
    if not inspect.isclass(object):
        return inspect.getfile(object)
    
    # Lookup by parent module (as in current inspect)
    if hasattr(object, '__module__'):
        object_ = sys.modules.get(object.__module__)
        if hasattr(object_, '__file__'):
            return object_.__file__
    
    # If parent module is __main__, lookup by methods (NEW)
    for _, member in inspect.getmembers(object):
        if inspect.isfunction(member) and object.__qualname__ + '.' + member.__name__ == member.__qualname__:
            return inspect.getfile(member)
    else:
        raise TypeError('Source for {!r} not found'.format(object))

def get_code_from_class(policy):
    klasses = inspect.getmro(policy)[-2::-1]
    whole_str = ""
    for klass in klasses:
        if is_interpreter_jupyter():
            cell_code = "".join(inspect.linecache.getlines(new_getfile(klass)))
            class_code = extract_symbols(cell_code, klass.__name__)[0][0]
        else:
            class_code = inspect.getsource(klass)
        whole_str += class_code
    return whole_str

def syft_function(
    input_policy: Union[InputPolicy, Any],
    input_policy_init_args: Dict[str, Any],
    output_policy: Union[OutputPolicy, Any],
    output_policy_init_args: Dict[str, Any],
    outputs: List[str]
) -> SubmitUserCode:
    # TODO: fix this for jupyter
    # TODO: add import validator
    
    if isinstance(input_policy, InputPolicy):
        input_policy = input_policy(input_policy_init_args)
        input_policy_init_args = None
    else:
          input_policy = SubmitUserPolicy(
            code='@serializable(recursive_serde=True)\n'+get_code_from_class(input_policy),
            class_name=input_policy.__name__,
            input_kwargs=input_policy.__init__.__code__.co_varnames[1:],
        )
    
    
    if isinstance(output_policy, OutputPolicy):
        output_policy = output_policy(output_policy_init_args)
        output_policy_init_args = None
    else:
        output_policy_submit = SubmitUserPolicy(
            code='@serializable(recursive_serde=True)\n'+get_code_from_class(output_policy),
            class_name=output_policy.__name__,
            input_kwargs=output_policy.__init__.__code__.co_varnames[1:],
        )
    
    def decorator(f):
        return SubmitUserCode(
            code=inspect.getsource(f),
            func_name=f.__name__,
            signature=inspect.signature(f),
            input_policy=input_policy,
            output_policy=output_policy_submit,
            local_function=f,
            input_kwargs=f.__code__.co_varnames,
            outputs=outputs,
            input_policy_init_args=input_policy_init_args,
            output_policy_init_args=output_policy_init_args
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
    input_kwargs: Dict[str, Any],
    outputs: List[str],
) -> str:
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
        return_annotation = ast.parse("Dict[str, Any]", mode="eval").body
    else:
        return_stmt = ast.Return(value=ast.Name(id="result"))
        return_annotation = ast.parse("Any", mode="eval").body

    new_body = tree.body + [call_stmt, return_stmt]

    wrapper_function = ast.FunctionDef(
        name=func_name,
        args=f.args,
        body=new_body,
        decorator_list=[],
        returns=return_annotation,
        lineno=0,
    )

    return ast.unparse(wrapper_function)


def new_check_code(context: TransformContext) -> TransformContext:
    try:
        processed_code = process_code(
            raw_code=context.output["raw_code"],
            func_name=context.output["unique_func_name"],
            original_func_name=context.output["service_func_name"],
            input_kwargs=context.output["input_kwargs"],
            outputs=context.output["outputs"],
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


def init_output_policy_state(context: TransformContext) -> TransformContext:
    if isinstance(context.output["output_policy"], OutputPolicy):
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
