# future
from __future__ import annotations

# stdlib
import ast
from enum import Enum
import hashlib
import inspect
from io import StringIO
import itertools
import sys
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from typing_extensions import Self

# relative
from ...abstract_node import NodeType
from ...client.api import NodeView
from ...client.enclave_client import EnclaveMetadata
from ...node.credentials import SyftVerifyKey
from ...serde.deserialize import _deserialize
from ...serde.serializable import serializable
from ...serde.serialize import _serialize
from ...store.document_store import PartitionKey
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftHashableObject
from ...types.syft_object import SyftObject
from ...types.transforms import TransformContext
from ...types.transforms import add_node_uid_for_key
from ...types.transforms import generate_id
from ...types.transforms import transform
from ...types.uid import UID
from ...util import options
from ...util.colors import SURFACE
from ...util.markdown import CodeMarkdown
from ...util.markdown import as_markdown_code
from ..context import AuthedServiceContext
from ..dataset.dataset import Asset
from ..policy.policy import CustomInputPolicy
from ..policy.policy import CustomOutputPolicy
from ..policy.policy import ExactMatch
from ..policy.policy import InputPolicy
from ..policy.policy import OutputPolicy
from ..policy.policy import Policy
from ..policy.policy import SingleExecutionExactOutput
from ..policy.policy import SubmitUserPolicy
from ..policy.policy import UserPolicy
from ..policy.policy import init_policy
from ..policy.policy import load_policy_code
from ..policy.policy_service import PolicyService
from ..response import SyftError
from .code_parse import GlobalsVisitor
from .unparse import unparse

UserVerifyKeyPartitionKey = PartitionKey(key="user_verify_key", type_=SyftVerifyKey)
CodeHashPartitionKey = PartitionKey(key="code_hash", type_=int)

PyCodeObject = Any


def extract_uids(kwargs: Dict[str, Any]) -> Dict[str, UID]:
    # relative
    from ...types.twin_object import TwinObject
    from ..action.action_object import ActionObject

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


@serializable()
class UserCodeStatus(Enum):
    SUBMITTED = "submitted"
    DENIED = "denied"
    EXECUTE = "execute"

    def __hash__(self) -> int:
        return hash(self.value)


# User Code status context for multiple approvals
# To make nested dicts hashable for mongodb
# as status is in attr_searchable
@serializable(attrs=["base_dict"])
class UserCodeStatusContext(SyftHashableObject):
    base_dict: Dict = {}

    def __init__(self, base_dict: Dict):
        self.base_dict = base_dict

    def __repr__(self):
        return str(self.base_dict)

    def _repr_html_(self):
        string = f"""
            <style>
                .syft-user_code {{color: {SURFACE[options.color_theme]};}}
                </style>
                <div class='syft-user_code'>
                    <h3 style="line-height: 25%; margin-top: 25px;">User Code Status</h3>
                    <p style="margin-left: 3px;">
            """
        for node_view, status in self.base_dict.items():
            node_name_str = f"{node_view.node_name}"
            uid_str = f"{node_view.node_id}"
            status_str = f"{status.value}"

            string += f"""
                    &#x2022; <strong>UID: </strong>{uid_str}&nbsp;
                    <strong>Node name: </strong>{node_name_str}&nbsp;
                    <strong>Status: </strong>{status_str}
                    <br>
                """
        string += "</p></div>"
        return string

    def __repr_syft_nested__(self):
        string = ""
        for node_view, status in self.base_dict.items():
            string += f"{node_view.node_name}: {status}<br>"
        return string

    @property
    def approved(self) -> bool:
        # approved for this node only
        statuses = set(self.base_dict.values())
        return len(statuses) == 1 and UserCodeStatus.EXECUTE in statuses

    def for_context(self, context: AuthedServiceContext) -> UserCodeStatus:
        if context.node.node_type == NodeType.ENCLAVE:
            keys = set(self.base_dict.values())
            if len(keys) == 1 and UserCodeStatus.EXECUTE in keys:
                return UserCodeStatus.EXECUTE
            elif UserCodeStatus.SUBMITTED in keys and UserCodeStatus.DENIED not in keys:
                return UserCodeStatus.SUBMITTED
            elif UserCodeStatus.DENIED in keys:
                return UserCodeStatus.DENIED
            else:
                return Exception(f"Invalid types in {keys} for Code Submission")

        elif context.node.node_type == NodeType.DOMAIN:
            node_view = NodeView(
                node_name=context.node.name,
                node_id=context.node.id,
                verify_key=context.node.signing_key.verify_key,
            )
            if node_view in self.base_dict:
                return self.base_dict[node_view]
            else:
                raise Exception(
                    f"Code Object does not contain {context.node.name} Domain's data"
                )
        else:
            raise Exception(
                f"Invalid Node Type for Code Submission:{context.node.node_type}"
            )

    def mutate(
        self, value: UserCodeStatus, node_name: str, node_id, verify_key: SyftVerifyKey
    ) -> Union[SyftError, Self]:
        node_view = NodeView(
            node_name=node_name, node_id=node_id, verify_key=verify_key
        )
        base_dict = self.base_dict
        if node_view in base_dict:
            base_dict[node_view] = value
            self.base_dict = base_dict
            return self
        else:
            return SyftError(
                message="Cannot Modify Status as the Domain's data is not included in the request"
            )


@serializable()
class UserCode(SyftObject):
    # version
    __canonical_name__ = "UserCode"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    node_uid: Optional[UID]
    user_verify_key: SyftVerifyKey
    raw_code: str
    input_policy_type: Union[Type[InputPolicy], UserPolicy]
    input_policy_init_kwargs: Optional[Dict[Any, Any]] = None
    input_policy_state: bytes = b""
    output_policy_type: Union[Type[OutputPolicy], UserPolicy]
    output_policy_init_kwargs: Optional[Dict[Any, Any]] = None
    output_policy_state: bytes = b""
    parsed_code: str
    service_func_name: str
    unique_func_name: str
    user_unique_func_name: str
    code_hash: str
    signature: inspect.Signature
    status: UserCodeStatusContext
    input_kwargs: List[str]
    enclave_metadata: Optional[EnclaveMetadata] = None

    __attr_searchable__ = ["user_verify_key", "status", "service_func_name"]
    __attr_unique__ = ["code_hash", "user_unique_func_name"]
    __repr_attrs__ = ["status.approved", "service_func_name", "shareholders"]

    def __setattr__(self, key: str, value: Any) -> None:
        attr = getattr(type(self), key, None)
        if inspect.isdatadescriptor(attr):
            attr.fset(self, value)
        else:
            return super().__setattr__(key, value)

    def _coll_repr_(self) -> Dict[str, Any]:
        status = list(self.status.base_dict.values())[0].value
        if status == UserCodeStatus.SUBMITTED.value:
            badge_color = "badge-purple"
        elif status == UserCodeStatus.EXECUTE.value:
            badge_color = "badge-green"
        else:
            badge_color = "badge-red"
        status_badge = {"value": status, "type": badge_color}
        return {
            "Input Policy": self.input_policy_type.__canonical_name__,
            "Output Policy": self.output_policy_type.__canonical_name__,
            "Function name": self.service_func_name,
            "User verify key": {
                "value": str(self.user_verify_key),
                "type": "clipboard",
            },
            "Status": status_badge,
        }

    @property
    def shareholders(self) -> List[str]:
        node_names_list = []
        nodes = self.input_policy_init_kwargs.keys()
        for node_view in nodes:
            node_names_list.append(str(node_view.node_name))
        return node_names_list

    @property
    def input_policy(self) -> Optional[InputPolicy]:
        if not self.status.approved:
            return None

        if len(self.input_policy_state) == 0:
            input_policy = None
            if isinstance(self.input_policy_type, type) and issubclass(
                self.input_policy_type, InputPolicy
            ):
                # TODO: Tech Debt here
                node_view_workaround = False
                for k, _ in self.input_policy_init_kwargs.items():
                    if isinstance(k, NodeView):
                        node_view_workaround = True

                if node_view_workaround:
                    input_policy = self.input_policy_type(
                        init_kwargs=self.input_policy_init_kwargs
                    )
                else:
                    input_policy = self.input_policy_type(
                        **self.input_policy_init_kwargs
                    )
            elif isinstance(self.input_policy_type, UserPolicy):
                input_policy = init_policy(
                    self.input_policy_type, self.input_policy_init_kwargs
                )
            else:
                raise Exception(f"Invalid output_policy_type: {self.input_policy_type}")

            if input_policy is not None:
                input_blob = _serialize(input_policy, to_bytes=True)
                self.input_policy_state = input_blob
                return input_policy
            else:
                raise Exception("input_policy is None during init")
        try:
            return _deserialize(self.input_policy_state, from_bytes=True)
        except Exception as e:
            print(f"Failed to deserialize custom input policy state. {e}")
            return None

    @property
    def output_policy(self) -> Optional[OutputPolicy]:
        if not self.status.approved:
            return None

        if len(self.output_policy_state) == 0:
            output_policy = None
            if isinstance(self.output_policy_type, type) and issubclass(
                self.output_policy_type, OutputPolicy
            ):
                output_policy = self.output_policy_type(
                    **self.output_policy_init_kwargs
                )
            elif isinstance(self.output_policy_type, UserPolicy):
                output_policy = init_policy(
                    self.output_policy_type, self.output_policy_init_kwargs
                )
            else:
                raise Exception(
                    f"Invalid output_policy_type: {self.output_policy_type}"
                )

            if output_policy is not None:
                output_blob = _serialize(output_policy, to_bytes=True)
                self.output_policy_state = output_blob
                return output_policy
            else:
                raise Exception("output_policy is None during init")

        try:
            return _deserialize(self.output_policy_state, from_bytes=True)
        except Exception as e:
            print(f"Failed to deserialize custom output policy state. {e}")
            return None

    @input_policy.setter
    def input_policy(self, value: Any) -> None:
        if isinstance(value, InputPolicy):
            self.input_policy_state = _serialize(value, to_bytes=True)
        elif (isinstance(value, bytes) and len(value) == 0) or value is None:
            self.input_policy_state = b""
        else:
            raise Exception(f"You can't set {type(value)} as input_policy_state")

    @output_policy.setter
    def output_policy(self, value: Any) -> None:
        if isinstance(value, OutputPolicy):
            self.output_policy_state = _serialize(value, to_bytes=True)
        elif (isinstance(value, bytes) and len(value) == 0) or value is None:
            self.output_policy_state = b""
        else:
            raise Exception(f"You can't set {type(value)} as output_policy_state")

    @property
    def byte_code(self) -> Optional[PyCodeObject]:
        return compile_byte_code(self.parsed_code)

    @property
    def assets(self) -> List[Asset]:
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(self.node_uid, self.syft_client_verify_key)
        if api is None:
            return SyftError(message=f"You must login to {self.node_uid}")

        inputs = (
            uids
            for node_view, uids in self.input_policy_init_kwargs.items()
            if node_view.node_name == api.node_name
        )
        all_assets = []
        for uid in itertools.chain.from_iterable(x.values() for x in inputs):
            if isinstance(uid, UID):
                assets = api.services.dataset.get_assets_by_action_id(uid)
                if not isinstance(assets, list):
                    return assets

                all_assets += assets
        return all_assets

    @property
    def unsafe_function(self) -> Optional[Callable]:
        print("WARNING: This code was submitted by a User and could be UNSAFE.")

        # 🟡 TODO: re-use the same infrastructure as the execute_byte_code function
        def wrapper(*args: Any, **kwargs: Any) -> Callable:
            try:
                filtered_kwargs = {}
                on_private_data, on_mock_data = False, False
                for k, v in kwargs.items():
                    filtered_kwargs[k], arg_type = debox_asset(v)
                    on_private_data = (
                        on_private_data or arg_type == ArgumentType.PRIVATE
                    )
                    on_mock_data = on_mock_data or arg_type == ArgumentType.MOCK
                if on_private_data:
                    print("Warning: The result you see is computed on PRIVATE data.")
                elif on_mock_data:
                    print("Warning: The result you see is computed on MOCK data.")

                # remove the decorator
                inner_function = ast.parse(self.raw_code).body[0]
                inner_function.decorator_list = []
                # compile the function
                raw_byte_code = compile_byte_code(unparse(inner_function))
                # load it
                exec(raw_byte_code)  # nosec
                # execute it
                evil_string = f"{self.service_func_name}(**filtered_kwargs)"
                result = eval(evil_string, None, locals())  # nosec
                # return the results
                return result
            except Exception as e:
                print(f"Failed to run unsafe_function. {e}")

        return wrapper

    def _repr_markdown_(self):
        md = f"""class UserCode
    id: UID = {self.id}
    status.approved: bool = {self.status.approved}
    service_func_name: str = {self.service_func_name}
    shareholders: list = {self.shareholders}
    code:

{self.raw_code}"""
        return as_markdown_code(md)

    @property
    def show_code(self) -> CodeMarkdown:
        return CodeMarkdown(self.raw_code)

    def show_code_cell(self):
        warning_message = """# WARNING: \n# Before you submit
# change the name of the function \n# for no duplicates\n\n"""

        # third party
        from IPython import get_ipython

        ip = get_ipython()
        ip.set_next_input(warning_message + self.raw_code)


@serializable(without=["local_function"])
class SubmitUserCode(SyftObject):
    # version
    __canonical_name__ = "SubmitUserCode"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    code: str
    func_name: str
    signature: inspect.Signature
    input_policy_type: Union[SubmitUserPolicy, UID, Type[InputPolicy]]
    input_policy_init_kwargs: Optional[Dict[Any, Any]] = {}
    output_policy_type: Union[SubmitUserPolicy, UID, Type[OutputPolicy]]
    output_policy_init_kwargs: Optional[Dict[Any, Any]] = {}
    local_function: Optional[Callable]
    input_kwargs: List[str]
    enclave_metadata: Optional[EnclaveMetadata] = None

    __repr_attrs__ = ["func_name", "code"]

    @property
    def kwargs(self) -> List[str]:
        return self.input_policy_init_kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # only run this on the client side
        if self.local_function:
            # filtered_args = []
            filtered_kwargs = {}
            # for arg in args:
            #     filtered_args.append(debox_asset(arg))
            on_private_data, on_mock_data = False, False
            for k, v in kwargs.items():
                filtered_kwargs[k], arg_type = debox_asset(v)
                on_private_data = on_private_data or arg_type == ArgumentType.PRIVATE
                on_mock_data = on_mock_data or arg_type == ArgumentType.MOCK
            if on_private_data:
                print("Warning: The result you see is computed on PRIVATE data.")
            elif on_mock_data:
                print("Warning: The result you see is computed on MOCK data.")
            return self.local_function(**filtered_kwargs)
        else:
            raise NotImplementedError


class ArgumentType(Enum):
    REAL = 1
    MOCK = 2
    PRIVATE = 4


def debox_asset(arg: Any) -> Any:
    deboxed_arg = arg
    if isinstance(deboxed_arg, Asset):
        asset = deboxed_arg
        if asset.has_data_permission():
            return asset.data, ArgumentType.PRIVATE
        else:
            return asset.mock, ArgumentType.MOCK
    if hasattr(deboxed_arg, "syft_action_data"):
        deboxed_arg = deboxed_arg.syft_action_data
    return deboxed_arg, ArgumentType.REAL


def syft_function_single_use(*args: Any, **kwargs: Any):
    return syft_function(
        input_policy=ExactMatch(*args, **kwargs),
        output_policy=SingleExecutionExactOutput(),
    )


def syft_function(
    input_policy: Union[InputPolicy, UID],
    output_policy: Optional[Union[OutputPolicy, UID]] = None,
) -> SubmitUserCode:
    if isinstance(input_policy, CustomInputPolicy):
        input_policy_type = SubmitUserPolicy.from_obj(input_policy)
    else:
        input_policy_type = type(input_policy)

    if output_policy is None:
        output_policy = SingleExecutionExactOutput()

    if isinstance(output_policy, CustomOutputPolicy):
        output_policy_type = SubmitUserPolicy.from_obj(output_policy)
    else:
        output_policy_type = type(output_policy)

    def decorator(f):
        print(
            f"Syft function '{f.__name__}' successfully created. "
            f"To add a code request, please create a project using `project = syft.Project(...)`, "
            f"then use command `project.create_code_request`."
        )
        return SubmitUserCode(
            code=inspect.getsource(f),
            func_name=f.__name__,
            signature=inspect.signature(f),
            input_policy_type=input_policy_type,
            input_policy_init_kwargs=input_policy.init_kwargs,
            output_policy_type=output_policy_type,
            output_policy_init_kwargs=output_policy.init_kwargs,
            local_function=f,
            input_kwargs=f.__code__.co_varnames[: f.__code__.co_argcount],
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
    input_kwargs: List[str],
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

    return_stmt = ast.Return(value=ast.Name(id="result"))
    new_body = tree.body + [call_stmt, return_stmt]

    wrapper_function = ast.FunctionDef(
        name=func_name,
        args=f.args,
        body=new_body,
        decorator_list=[],
        returns=None,
        lineno=0,
    )

    return unparse(wrapper_function)


def new_check_code(context: TransformContext) -> TransformContext:
    # TODO remove this tech debt hack
    input_kwargs = context.output["input_policy_init_kwargs"]
    node_view_workaround = False
    for k in input_kwargs.keys():
        if isinstance(k, NodeView):
            node_view_workaround = True

    if not node_view_workaround:
        input_keys = list(input_kwargs.keys())
    else:
        input_keys = []
        for d in input_kwargs.values():
            input_keys += d.keys()

    processed_code = process_code(
        raw_code=context.output["raw_code"],
        func_name=context.output["unique_func_name"],
        original_func_name=context.output["service_func_name"],
        input_kwargs=input_keys,
    )
    context.output["parsed_code"] = processed_code

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


def check_policy(policy: Policy, context: TransformContext) -> TransformContext:
    policy_service = context.node.get_service(PolicyService)
    if isinstance(policy, SubmitUserPolicy):
        policy = policy.to(UserPolicy, context=context)
    elif isinstance(policy, UID):
        policy = policy_service.get_policy_by_uid(context, policy)
        if policy.is_ok():
            policy = policy.ok()

    return policy


def check_input_policy(context: TransformContext) -> TransformContext:
    ip = context.output["input_policy_type"]
    ip = check_policy(policy=ip, context=context)
    context.output["input_policy_type"] = ip
    return context


def check_output_policy(context: TransformContext) -> TransformContext:
    op = context.output["output_policy_type"]
    op = check_policy(policy=op, context=context)
    context.output["output_policy_type"] = op
    return context


def add_custom_status(context: TransformContext) -> TransformContext:
    input_keys = list(context.output["input_policy_init_kwargs"].keys())
    if context.node.node_type == NodeType.DOMAIN:
        node_view = NodeView(
            node_name=context.node.name,
            node_id=context.node.id,
            verify_key=context.node.signing_key.verify_key,
        )
        context.output["status"] = UserCodeStatusContext(
            base_dict={node_view: UserCodeStatus.SUBMITTED}
        )
        # if node_view in input_keys or len(input_keys) == 0:
        #     context.output["status"] = UserCodeStatusContext(
        #         base_dict={node_view: UserCodeStatus.SUBMITTED}
        #     )
        # else:
        #     raise ValueError(f"Invalid input keys: {input_keys} for {node_view}")
    elif context.node.node_type == NodeType.ENCLAVE:
        base_dict = {key: UserCodeStatus.SUBMITTED for key in input_keys}
        context.output["status"] = UserCodeStatusContext(base_dict=base_dict)
    else:
        raise NotImplementedError(
            f"Invalid node type:{context.node.node_type} for code submission"
        )
    return context


@transform(SubmitUserCode, UserCode)
def submit_user_code_to_user_code() -> List[Callable]:
    return [
        generate_id,
        hash_code,
        generate_unique_func_name,
        check_input_policy,
        check_output_policy,
        new_check_code,
        add_credentials_for_key("user_verify_key"),
        add_custom_status,
        add_node_uid_for_key("node_uid"),
    ]


@serializable()
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


def load_approved_policy_code(user_code_items: List[UserCode]) -> Any:
    """Reload the policy code in memory for user code that is approved."""
    try:
        for user_code in user_code_items:
            if user_code.status.approved:
                if isinstance(user_code.input_policy_type, UserPolicy):
                    load_policy_code(user_code.input_policy_type)
                if isinstance(user_code.output_policy_type, UserPolicy):
                    load_policy_code(user_code.output_policy_type)
    except Exception as e:
        raise Exception(f"Failed to load code: {user_code}: {e}")
