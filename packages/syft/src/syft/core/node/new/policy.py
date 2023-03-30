# future
from __future__ import annotations

# stdlib
import ast
from copy import deepcopy
from enum import Enum
import hashlib
import inspect
from inspect import Parameter
from inspect import Signature
from io import StringIO
import sys
import types
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
from RestrictedPython import compile_restricted
from result import Ok
from result import Result

# relative
from ....util import is_interpreter_jupyter
from .action_object import ActionObject
from .api import NodeView
from .code_parse import GlobalsVisitor
from .context import AuthedServiceContext
from .context import NodeServiceContext
from .credentials import SyftVerifyKey
from .dataset import Asset
from .datetime import DateTime
from .deserialize import _deserialize
from .document_store import PartitionKey
from .node import NodeType
from .response import SyftError
from .response import SyftSuccess
from .serializable import serializable
from .serialize import _serialize
from .syft_object import SYFT_OBJECT_VERSION_1
from .syft_object import SyftObject
from .transforms import TransformContext
from .transforms import generate_id
from .transforms import transform
from .twin_object import TwinObject
from .uid import UID
from .unparse import unparse

PolicyUserVerifyKeyPartitionKey = PartitionKey(
    key="user_verify_key", type_=SyftVerifyKey
)
PyCodeObject = Any


def extract_uid(v: Any) -> UID:
    value = v
    if isinstance(v, ActionObject):
        value = v.id
    if isinstance(v, TwinObject):
        value = v.id

    if not isinstance(value, UID):
        raise Exception(f"Input {v} must have a UID not {type(v)}")
    return value


def filter_only_uids(results: Any) -> Dict[str, Any]:
    if not hasattr(results, "__len__"):
        results = [results]

    if isinstance(results, list):
        output_list = []
        for v in results:
            output_list.append(extract_uid(v))
        return output_list
    elif isinstance(results, dict):
        output_dict = {}
        for k, v in results.items():
            output_dict[k] = extract_uid(v)
        return output_dict
    return extract_uid(results)


class Policy(SyftObject):
    # version
    __canonical_name__ = "Policy"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    init_kwargs: Dict[Any, Any] = {}

    def __init__(self, *args, **kwargs) -> None:
        if "init_kwargs" in kwargs:
            init_kwargs = kwargs["init_kwargs"]
            del kwargs["init_kwargs"]
        else:
            init_kwargs = deepcopy(kwargs)
            if "id" in init_kwargs:
                del init_kwargs["id"]
        super().__init__(init_kwargs=init_kwargs, *args, **kwargs)

    @property
    def policy_code(self) -> str:
        cls = type(self)
        op_code = inspect.getsource(cls)
        return op_code

    def public_state() -> None:
        raise NotImplementedError

    @property
    def valid(self) -> Union[SyftSuccess, SyftError]:
        return SyftSuccess(message="Policy is valid.")


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


class InputPolicy(Policy):
    # version
    __canonical_name__ = "InputPolicy"
    __version__ = SYFT_OBJECT_VERSION_1

    # id: UID
    # input_kwargs: Dict[NodeView, Any]
    # node_uid: Optional[UID]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # TODO: This method initialization would conflict if one of the input variables
        # to the code submission function happens to be id or input_kwargs
        # uid = UID()
        # node_uid = None
        # if "id" in kwargs:
        #     uid = kwargs["id"]
        # if "node_uid" in kwargs:
        #     node_uid = kwargs["node_uid"]

        # # finally get inputs
        print("getting input policy args and kwargs", args, kwargs)
        if "init_kwargs" in kwargs:
            init_kwargs = kwargs["init_kwargs"]
            del kwargs["init_kwargs"]
        else:
            print("partition by node")
            init_kwargs = partition_by_node(kwargs)
        print("after and kwargs", args, kwargs)
        super().__init__(*args, init_kwargs=init_kwargs, **kwargs)
        # super().__init__(id=uid, input_kwargs=kwargs, node_uid=node_uid)

    def filter_kwargs(
        self, kwargs: Dict[str, Any], context: AuthedServiceContext, code_item_id: UID
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def inputs(self) -> Dict[NodeView, Any]:
        return self.init_kwargs

    # @property
    # def assets(self) -> List[Asset]:
    #     # relative
    #     from .api import APIRegistry

    #     api = APIRegistry.api_for(self.node_uid)
    #     if api is None:
    #         return SyftError(message=f"You must login to {self.node_uid}")

    #     node_view = NodeView(
    #         node_name=api.node_name, verify_key=api.signing_key.verify_key
    #     )
    #     inputs = self.inputs[node_view]
    #     all_assets = []
    #     for k, uid in inputs.items():
    #         if isinstance(uid, UID):
    #             assets = api.services.dataset.get_assets_by_action_id(uid)
    #             if not isinstance(assets, list):
    #                 return assets

    #             all_assets += assets
    #     return all_assets


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


class OutputPolicy(Policy):
    # version
    __canonical_name__ = "OutputPolicy"
    __version__ = SYFT_OBJECT_VERSION_1

    output_history: List[OutputHistory] = []
    output_kwargs: List[str] = []
    node_uid: Optional[UID]

    def apply_output(
        self,
        context: NodeServiceContext,
        outputs: Any,
    ) -> Any:
        output_uids = filter_only_uids(outputs)
        if isinstance(output_uids, UID):
            output_uids = [output_uids]
        history = OutputHistory(
            output_time=DateTime.now(),
            outputs=output_uids,
            executing_user_verify_key=context.credentials,
        )
        self.output_history.append(history)
        return outputs

    @property
    def outputs(self) -> List[str]:
        return self.output_kwargs


@serializable()
class OutputPolicyExecuteCount(OutputPolicy):
    __canonical_name__ = "OutputPolicyExecuteCount"
    __version__ = SYFT_OBJECT_VERSION_1

    count: int = 0
    limit: int

    def apply_output(
        self,
        context: NodeServiceContext,
        outputs: Any,
    ) -> Optional[Any]:
        if self.count < self.limit:
            super().apply_output(context, outputs)
            self.count += 1
            return outputs
        return None

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

    def public_state(self) -> None:
        return {"limit": self.limit, "count": self.count}


@serializable()
class OutputPolicyExecuteOnce(OutputPolicyExecuteCount):
    __canonical_name__ = "OutputPolicyExecuteOnce"
    __version__ = SYFT_OBJECT_VERSION_1

    limit: int = 1


SingleExecutionExactOutput = OutputPolicyExecuteOnce


class CustomPolicy(Policy):
    # version
    __canonical_name__ = "CustomPolicy"
    __version__ = SYFT_OBJECT_VERSION_1


class CustomOutputPolicy(CustomPolicy, OutputPolicy):
    # version
    __canonical_name__ = "CustomOutputPolicy"
    __version__ = SYFT_OBJECT_VERSION_1

    def apply_output(
        self,
        context: NodeServiceContext,
        outputs: Any,
    ) -> Optional[Any]:
        return outputs


class CustomInputPolicy(CustomPolicy, InputPolicy):
    # version
    __canonical_name__ = "CustomInputPolicy"
    __version__ = SYFT_OBJECT_VERSION_1


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
    policy_version: int

    @property
    def byte_code(self) -> Optional[PyCodeObject]:
        return compile_byte_code(self.parsed_code)

    @property
    def policy_code(self) -> str:
        return self.raw_code

    def apply_output(
        self,
        context: NodeServiceContext,
        outputs: Any,
    ) -> Optional[Any]:
        return outputs


def new_getfile(object):
    if not inspect.isclass(object):
        return inspect.getfile(object)

    # Lookup by parent module (as in current inspect)
    if hasattr(object, "__module__"):
        object_ = sys.modules.get(object.__module__)
        if hasattr(object_, "__file__"):
            return object_.__file__

    # If parent module is __main__, lookup by methods (NEW)
    for _, member in inspect.getmembers(object):
        if (
            inspect.isfunction(member)
            and object.__qualname__ + "." + member.__name__ == member.__qualname__
        ):
            return inspect.getfile(member)
    else:
        raise TypeError("Source for {!r} not found".format(object))


def get_code_from_class(policy):
    klasses = [inspect.getmro(policy)[0]]  #
    whole_str = ""
    for klass in klasses:
        if is_interpreter_jupyter():
            # third party
            from IPython.core.magics.code import extract_symbols

            cell_code = "".join(inspect.linecache.getlines(new_getfile(klass)))
            class_code = extract_symbols(cell_code, klass.__name__)[0][0]
        else:
            class_code = inspect.getsource(klass)
        whole_str += class_code
    return whole_str


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

    @staticmethod
    def from_obj(policy_obj: CustomPolicy) -> SubmitUserPolicy:
        user_class = policy_obj.__class__
        init_f_code = user_class.__init__.__code__
        return SubmitUserPolicy(
            code=get_code_from_class(user_class),
            class_name=user_class.__name__,
            input_kwargs=init_f_code.co_varnames[1 : init_f_code.co_argcount],
        )


def hash_code(context: TransformContext) -> TransformContext:
    code = context.output["code"]
    del context.output["code"]
    context.output["raw_code"] = code
    code_hash = hashlib.sha256(code.encode("utf8")).hexdigest()
    context.output["code_hash"] = code_hash
    return context


def generate_unique_class_name(context: TransformContext) -> TransformContext:
    # TODO: Do we need to check if the initial name contains underscores?
    code_hash = context.output["code_hash"]
    service_class_name = context.output["class_name"]
    unique_name = f"{service_class_name}_{context.credentials}_{code_hash}"
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

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ClassDef):
        raise Exception(
            "Class code should only contain the Class Definition for your policy."
        )

    old_class = tree.body[0]
    if len(old_class.bases) != 1 or old_class.bases[0].id not in [
        "CustomInputPolicy",
        "CustomOutputPolicy",
    ]:
        raise Exception(
            "Class code should either implement CustomInputPolicy or CustomOutputPolicy"
        )

    # TODO: changes the bases

    serializable_name = ast.Name(id="serializable", ctx=ast.Load())
    serializable_decorator = ast.Call(
        func=serializable_name,
        args=[],
        keywords=[],
    )

    new_class = tree.body[0]
    version = None
    # TODO add this manually
    for stmt in new_class.body:
        if isinstance(stmt, ast.Assign):
            if stmt.targets[0].id == "__version__":
                version = stmt.value.value
                break
    if version is None:
        raise Exception(
            "Version cannot be found. Please specify it in your custom policy"
        )

    # change the module that the code will reference
    # this is required for the @serializable to mount it in the right path for serde
    new_line = ast.parse(f"__module__ = 'syft.user'")
    new_class.body.append(new_line.body[0])

    new_class.name = class_name
    new_class.decorator_list = [serializable_decorator]
    return unparse(new_class), version


def check_class_code(context: TransformContext) -> TransformContext:
    # TODO: define the proper checking for this case based on the ideas from UserCode
    # check for no globals
    # check for Policy template -> __init__, apply_output, public_state
    # parse init signature
    # check dangerous libraries, maybe compile_restricted already does that
    try:
        processed_code, policy_version = process_class_code(
            raw_code=context.output["raw_code"],
            class_name=context.output["unique_name"],
            input_kwargs=context.output["input_kwargs"],
        )
        context.output["parsed_code"] = processed_code
        context.output["policy_version"] = policy_version
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


@transform(SubmitUserPolicy, UserPolicy)
def submit_policy_code_to_user_code() -> List[Callable]:
    return [
        generate_id,
        hash_code,
        generate_unique_class_name,
        generate_signature,
        check_class_code,
        # compile_code, # don't compile until approved
        add_credentials_for_key("user_verify_key"),
    ]


def add_class_to_user_module(klass: type, unique_name: str) -> type:
    klass.__module__ = "syft.user"
    klass.__name__ = unique_name
    # syft absolute
    import syft as sy

    if not hasattr(sy, "user"):
        user_module = types.ModuleType("user")
        setattr(sys.modules["syft"], "user", user_module)
    user_module = sy.user
    setattr(user_module, unique_name, klass)
    setattr(sys.modules["syft"], "user", user_module)
    print("syft user module", sy.user, klass)
    return klass


def execute_policy_code(user_policy: UserPolicy):
    stdout_ = sys.stdout
    stderr_ = sys.stderr

    try:
        stdout = StringIO()
        stderr = StringIO()

        sys.stdout = stdout
        sys.stderr = stderr
        # syft absolute
        import syft as sy  # noqa: F401 # provide sy.Things to user code

        print("executing policy code", file=stderr_)
        print(
            "executing policy code unqiue name", user_policy.unique_name, file=stderr_
        )
        print(
            "executing policy code parsed_code", user_policy.parsed_code, file=stderr_
        )
        class_name = f"{user_policy.unique_name}_{user_policy.policy_version}"
        print("Exec context", file=stderr_)
        if class_name in user_policy.__object_version_registry__.keys():
            policy_class = user_policy.__object_version_registry__[class_name]
        else:
            exec(user_policy.byte_code)  # nosec
            policy_class = eval(user_policy.unique_name)  # nosec

        policy_class = add_class_to_user_module(policy_class, user_policy.unique_name)

        sys.stdout = stdout_
        sys.stderr = stderr_

        return policy_class

    except Exception as e:
        print("execute_byte_code failed", e, file=stderr_)

    finally:
        sys.stdout = stdout_
        sys.stderr = stderr_


# def import_policy(policy_class_name: str) -> Any:
#     user_policy =
#     policy_class = execute_policy_code(user_policy)
#     return policy_class


def load_policy_code(user_policy: UserPolicy) -> Any:
    try:
        print("calling load policy code", user_policy)
        policy_class = execute_policy_code(user_policy)
        print("finished loading", policy_class)
        # syft absolute
        import syft as sy

        a = getattr(sy, "user", None)
        print("syft.user", dir(a))
        return policy_class
    except Exception as e:
        print("exception loading code", e)


def init_policy(user_policy: UserPolicy, init_args: Dict[str, Any]):
    policy_class = load_policy_code(user_policy)
    print("init class with args", policy_class)
    print("init class with args", init_args)
    print("class mro", policy_class.mro())
    policy_object = policy_class(**init_args)
    return policy_object


def get_policy_object(user_policy: UserPolicy, state: str) -> Result[Any, str]:
    policy_class = execute_policy_code(user_policy)
    policy_object = _deserialize(state, from_bytes=True, class_type=policy_class)
    return policy_object


def update_policy_state(policy_object):
    return _serialize(policy_object, to_bytes=True)
