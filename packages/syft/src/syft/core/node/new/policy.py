# future
from __future__ import annotations

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

# third party
from RestrictedPython import compile_restricted
from result import Ok
from result import Result

# relative
from .api import NodeView
from .context import AuthedServiceContext
from .context import NodeServiceContext
from .credentials import SyftVerifyKey
from .dataset import Asset
from .datetime import DateTime
from .deserialize import _deserialize
from .document_store import PartitionKey
from .node import NodeType
from .policy_code_parse import GlobalsVisitor
from .response import SyftError
from .response import SyftSuccess
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


class Policy(SyftObject):
    __canonical_name__ = "Policy"
    __version__ = SYFT_OBJECT_VERSION_1


@serializable()
class UserPolicyStatus(Enum):
    SUBMITTED = "submitted"
    DENIED = "denied"
    APPROVED = "approved"


def partition_by_node(kwargs: Dict[str, Any]) -> Dict[str, UID]:
    # relative
    from .action_object import ActionObject
    from .api import APIRegistry
    from .api import NodeView
    from .twin_object import TwinObject

    # fetches the all the current api's connected
    api_list = APIRegistry.get_all_api()
    output_kwargs = {}
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

        _obj_exists = False
        for api in api_list:
            if api.services.action.exists(uid):
                node_view = NodeView.from_api(api)
                if node_view not in output_kwargs:
                    output_kwargs[node_view] = {k: uid}
                else:
                    output_kwargs[node_view].update({k: uid})

                _obj_exists = True
                break

        if not _obj_exists:
            raise Exception(f"Input data {k}:{uid} does not belong to any Domain")

    return output_kwargs


@serializable()
class InputPolicyState(SyftObject):
    # version
    __canonical_name__ = "InputPolicyState"
    __version__ = SYFT_OBJECT_VERSION_1


class InputPolicy(SyftObject):
    # version
    __canonical_name__ = "InputPolicy"
    __version__ = SYFT_OBJECT_VERSION_1

    # relative
    from .api import NodeView

    id: UID
    inputs: Dict[NodeView, Any]
    node_uid: Optional[UID]
    state_type: Optional[Type[InputPolicyState]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # TODO: This method initialization would conflict if one of the input variables
        # to the code submission function happens to be id or inputs
        uid = UID()
        node_uid = None
        state_type = None
        if "id" in kwargs:
            uid = kwargs["id"]
        if "node_uid" in kwargs:
            node_uid = kwargs["node_uid"]
        if "state_type" in kwargs:
            state_type = kwargs["state_type"]

        # finally get inputs
        if "inputs" in kwargs:
            kwargs = kwargs["inputs"]
        else:
            kwargs = partition_by_node(kwargs)
        super().__init__(
            id=uid, inputs=kwargs, node_uid=node_uid, state_type=state_type
        )

    def filter_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    # def __getitem__(self, key: Union[int, str]) -> Optional[SyftObject]:
    #     if isinstance(key, int):
    #         key = list(self.inputs.keys())[key]
    #     uid = self.inputs[key]
    #     # TODO Add NODE UID or LINK so we can resolve this object
    #     return uid

    @property
    def assets(self) -> List[Asset]:
        # relative
        from .api import APIRegistry

        api = APIRegistry.api_for(self.node_uid)
        if api is None:
            return SyftError(message=f"You must login to {self.node_uid}")

        node_view = NodeView(
            node_name=api.node_name, verify_key=api.signing_key.verify_key
        )
        inputs = self.inputs[node_view]
        all_assets = []
        for k, uid in inputs.items():
            if isinstance(uid, UID):
                assets = api.services.dataset.get_assets_by_action_id(uid)
                if not isinstance(assets, list):
                    return assets

                all_assets += assets
        return all_assets


def retrieve_from_db(
    code_item_id: UID, allowed_inputs: Dict[str, UID], context: AuthedServiceContext
) -> Dict:
    # relative
    from .action_service import TwinMode

    action_service = context.node.get_service("actionservice")
    code_inputs = {}

    if context.node.node_type == NodeType.DOMAIN:
        for var_name, arg_id in allowed_inputs.items():
            kwarg_value = action_service.get(
                context=context, uid=arg_id, twin_mode=TwinMode.NONE
            )
            if kwarg_value.is_err():
                return kwarg_value
            code_inputs[var_name] = kwarg_value.ok()

    elif context.node.node_type == NodeType.ENCLAVE:
        # TODO 🟣 Temporarily added skip permission arguments for enclave
        # until permissions are fully integrated
        dict_object = action_service.get(
            context=context, uid=code_item_id, skip_permission=True
        )
        if dict_object.is_err():
            return dict_object
        for value in dict_object.ok().base_dict.values():
            code_inputs.update(value)

    else:
        raise Exception(
            f"Invalid Node Type for Code Submission:{context.node.node_type}"
        )
    return Ok(code_inputs)


def allowed_ids_only(
    allowed_inputs: Dict[str, UID],
    kwargs: Dict[str, Any],
    context: AuthedServiceContext,
) -> Dict[str, UID]:
    if context.node.node_type == NodeType.DOMAIN:
        node_view = NodeView(
            node_name=context.node.name, verify_key=context.node.signing_key.verify_key
        )
        allowed_inputs = allowed_inputs[node_view]
    elif context.node.node_type == NodeType.ENCLAVE:
        base_dict = {}
        for key in allowed_inputs.values():
            base_dict.update(key)
        allowed_inputs = base_dict
    else:
        raise Exception(
            f"Invalid Node Type for Code Submission:{context.node.node_type}"
        )
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


@serializable()
class ExactMatch(InputPolicy):
    # version
    __canonical_name__ = "ExactMatch"
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


@serializable()
class OutputHistory(SyftObject):
    # version
    __canonical_name__ = "OutputHistory"
    __version__ = SYFT_OBJECT_VERSION_1

    output_time: DateTime
    outputs: Optional[Union[List[UID], Dict[str, UID]]]
    executing_user_verify_key: SyftVerifyKey


@serializable()
class OutputPolicyState(SyftObject):
    # version
    __canonical_name__ = "OutputPolicyState"
    __version__ = SYFT_OBJECT_VERSION_1

    output_history: List[OutputHistory] = []

    @property
    def valid(self) -> Union[SyftSuccess, SyftError]:
        raise NotImplementedError

    def update_state(
        self,
        context: NodeServiceContext,
        outputs: Optional[Union[UID, List[UID], Dict[str, UID]]],
    ) -> None:
        if isinstance(outputs, UID):
            outputs = [outputs]
        history = OutputHistory(
            output_time=DateTime.now(),
            outputs=outputs,
            executing_user_verify_key=context.credentials,
        )
        self.output_history.append(history)


@serializable()
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
        super().update_state(context=context, outputs=outputs)
        self.count += 1


@serializable()
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
    node_uid: Optional[UID]

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


class CustomOutputPolicy(OutputPolicy):
    # version
    __canonical_name__ = "CustomOutputPolicy"
    __version__ = SYFT_OBJECT_VERSION_1
    init_args: Dict[str, Any] = {}
    kwargs: Dict[str, Any] = {}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)
        self.init_args = args
        self.kwargs = kwargs


@serializable()
class SingleExecutionExactOutput(OutputPolicy):
    # version
    __canonical_name__ = "SingleExecutionExactOutput"
    __version__ = SYFT_OBJECT_VERSION_1

    state_type: Type[OutputPolicyState] = OutputPolicyStateExecuteOnce


@serializable()
class UserPolicy(Policy):
    __canonical_name__ = "UserPolicy"
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


@serializable()
class SubmitUserPolicy(Policy):
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


def process_class_code(raw_code: str, class_name: str, input_kwargs: List[str]) -> str:
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
    try:
        processed_code = process_class_code(
            raw_code=context.output["code"],
            class_name=context.output["unique_name"],
            input_kwargs=context.output["input_kwargs"],
        )
        context.output["parsed_code"] = processed_code

    except Exception as e:
        raise e
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
    params = [
        Parameter(name=k, kind=Parameter.POSITIONAL_OR_KEYWORD)
        for k in context.output["input_kwargs"]
    ]
    sig = Signature(parameters=params)
    context.output["signature"] = sig
    return context


def serialization_addon(context: TransformContext) -> TransformContext:
    policy_code = context.output["code"]
    context.output["code"] = "@serializable()\n" + policy_code
    return context


@transform(SubmitUserPolicy, UserPolicy)
def submit_policy_code_to_user_code() -> List[Callable]:
    return [
        generate_id,
        serialization_addon,
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
