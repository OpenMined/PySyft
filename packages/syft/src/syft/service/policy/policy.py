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

# relative
from ...abstract_node import NodeType
from ...client.api import NodeIdentity
from ...node.credentials import SyftVerifyKey
from ...serde.recursive_primitives import recursive_serde_register_type
from ...serde.serializable import serializable
from ...store.document_store import PartitionKey
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.transforms import TransformContext
from ...types.transforms import generate_id
from ...types.transforms import transform
from ...types.twin_object import TwinObject
from ...types.uid import UID
from ...util.util import is_interpreter_jupyter
from ..action.action_object import ActionObject
from ..code.code_parse import GlobalsVisitor
from ..code.unparse import unparse
from ..context import AuthedServiceContext
from ..context import ChangeContext
from ..context import NodeServiceContext
from ..dataset.dataset import Asset
from ..response import SyftError
from ..response import SyftSuccess

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
        super().__init__(init_kwargs=init_kwargs, *args, **kwargs)  # noqa: B026

    @classmethod
    @property
    def policy_code(cls) -> str:
        mro = reversed(cls.mro())
        op_code = ""
        for klass in mro:
            if "Policy" in klass.__name__:
                op_code += inspect.getsource(klass)
                op_code += "\n"
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
    from ...client.api import APIRegistry
    from ...client.api import NodeIdentity
    from ...types.twin_object import TwinObject
    from ..action.action_object import ActionObject

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
                node_identity = NodeIdentity.from_api(api)
                if node_identity not in output_kwargs:
                    output_kwargs[node_identity] = {k: uid}
                else:
                    output_kwargs[node_identity].update({k: uid})

                _obj_exists = True
                break

        if not _obj_exists:
            raise Exception(f"Input data {k}:{uid} does not belong to any Domain")

    return output_kwargs


class InputPolicy(Policy):
    __canonical_name__ = "InputPolicy"
    __version__ = SYFT_OBJECT_VERSION_1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "init_kwargs" in kwargs:
            init_kwargs = kwargs["init_kwargs"]
            del kwargs["init_kwargs"]
        else:
            # TODO: remove this tech debt, dont remove the id mapping functionality
            init_kwargs = partition_by_node(kwargs)
        super().__init__(*args, init_kwargs=init_kwargs, **kwargs)

    def filter_kwargs(
        self, kwargs: Dict[Any, Any], context: AuthedServiceContext, code_item_id: UID
    ) -> Dict[Any, Any]:
        raise NotImplementedError

    @property
    def inputs(self) -> Dict[NodeIdentity, Any]:
        return self.init_kwargs

    def _inputs_for_context(self, context: ChangeContext):
        user_node_view = NodeIdentity.from_change_context(context)
        inputs = self.inputs[user_node_view]

        root_context = AuthedServiceContext(
            node=context.node, credentials=context.approving_user_credentials
        ).as_root_context()

        action_service = context.node.get_service("actionservice")
        for var_name, uid in inputs.items():
            action_object = action_service.get(uid=uid, context=root_context)
            if action_object.is_err():
                return SyftError(message=action_object.err())
            action_object_value = action_object.ok()
            # resolve syft action data from blob store
            if isinstance(action_object_value, TwinObject):
                action_object_value.private_obj.syft_action_data  # noqa: B018
                action_object_value.mock_obj.syft_action_data  # noqa: B018
            elif isinstance(action_object_value, ActionObject):
                action_object_value.syft_action_data  # noqa: B018
            inputs[var_name] = action_object_value
        return inputs


def retrieve_from_db(
    code_item_id: UID, allowed_inputs: Dict[str, UID], context: AuthedServiceContext
) -> Dict:
    # relative
    from ...service.action.action_object import TwinMode

    action_service = context.node.get_service("actionservice")
    code_inputs = {}

    # When we are retrieving the code from the database, we need to use the node's
    # verify key as the credentials. This is because when we approve the code, we
    # we allow the private data to be used only for this specific code.
    # but we are not modifying the permissions of the private data

    root_context = AuthedServiceContext(
        node=context.node, credentials=context.node.verify_key
    )
    if context.node.node_type == NodeType.DOMAIN:
        for var_name, arg_id in allowed_inputs.items():
            kwarg_value = action_service._get(
                context=root_context,
                uid=arg_id,
                twin_mode=TwinMode.NONE,
                has_permission=True,
            )
            if kwarg_value.is_err():
                return SyftError(message=kwarg_value.err())
            code_inputs[var_name] = kwarg_value.ok()

    elif context.node.node_type == NodeType.ENCLAVE:
        dict_object = action_service.get(context=root_context, uid=code_item_id)
        if dict_object.is_err():
            return SyftError(message=dict_object.err())
        for value in dict_object.ok().syft_action_data.values():
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
        node_identity = NodeIdentity(
            node_name=context.node.name,
            node_id=context.node.id,
            verify_key=context.node.signing_key.verify_key,
        )
        allowed_inputs = allowed_inputs.get(node_identity, {})
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
        self, kwargs: Dict[Any, Any], context: AuthedServiceContext, code_item_id: UID
    ) -> Dict[Any, Any]:
        allowed_inputs = allowed_ids_only(
            allowed_inputs=self.inputs, kwargs=kwargs, context=context
        )
        results = retrieve_from_db(
            code_item_id=code_item_id, allowed_inputs=allowed_inputs, context=context
        )
        return results


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
    output_readers: List[SyftVerifyKey] = []

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

    @property
    def last_output_ids(self) -> List[str]:
        return self.output_history[-1].outputs


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


@serializable()
class CustomPolicy(type):
    # capture the init_kwargs transparently
    def __call__(cls, *args: Any, **kwargs: Any) -> None:
        obj = super().__call__(*args, **kwargs)
        obj.init_kwargs = kwargs
        return obj


recursive_serde_register_type(CustomPolicy)


@serializable()
class CustomOutputPolicy(metaclass=CustomPolicy):
    def apply_output(
        self,
        context: NodeServiceContext,
        outputs: Any,
    ) -> Optional[Any]:
        return outputs


class UserOutputPolicy(OutputPolicy):
    __canonical_name__ = "UserOutputPolicy"
    pass


class UserInputPolicy(InputPolicy):
    __canonical_name__ = "UserInputPolicy"
    pass


class EmpyInputPolicy(InputPolicy):
    __canonical_name__ = "EmptyInputPolicy"
    pass


class CustomInputPolicy(metaclass=CustomPolicy):
    pass


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
        raise TypeError(f"Source for {object!r} not found")


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


def process_class_code(raw_code: str, class_name: str) -> str:
    tree = ast.parse(raw_code)
    v = GlobalsVisitor()
    v.visit(tree)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ClassDef):
        raise Exception(
            "Class code should only contain the Class definition for your policy"
        )
    old_class = tree.body[0]
    if len(old_class.bases) != 1 or old_class.bases[0].attr not in [
        CustomInputPolicy.__name__,
        CustomOutputPolicy.__name__,
    ]:
        raise Exception(
            f"Class code should either implement {CustomInputPolicy.__name__} "
            f"or {CustomOutputPolicy.__name__}"
        )

    # TODO: changes the bases
    old_class.bases[0].attr = old_class.bases[0].attr.replace("Custom", "User")

    serializable_name = ast.Name(id="sy.serializable", ctx=ast.Load())
    serializable_decorator = ast.Call(
        func=serializable_name,
        args=[],
        keywords=[],
    )

    new_class = tree.body[0]
    # TODO add this manually
    for stmt in new_class.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
            stmt.name = "__user_init__"

    # change the module that the code will reference
    # this is required for the @serializable to mount it in the right path for serde
    new_line = ast.parse('__module__ = "syft.user"')
    new_class.body.append(new_line.body[0])
    new_line = ast.parse(f'__canonical_name__ = "{class_name}"')
    new_class.body.append(new_line.body[0])
    new_line = ast.parse("__version__ = 1")
    new_class.body.append(new_line.body[0])
    new_class.name = class_name
    new_class.decorator_list = [serializable_decorator]
    new_body = []
    new_body.append(
        ast.ImportFrom(
            module="__future__",
            names=[ast.alias(name="annotations", asname="annotations")],
            level=0,
        )
    )
    new_body.append(ast.Import(names=[ast.alias(name="syft", asname="sy")], level=0))
    typing_types = [
        "Any",
        "Callable",
        "ClassVar",
        "Dict",
        "List",
        "Optional",
        "Set",
        "Tuple",
        "Type",
    ]
    for typing_type in typing_types:
        new_body.append(
            ast.ImportFrom(
                module="typing",
                names=[ast.alias(name=typing_type, asname=typing_type)],
                level=0,
            )
        )
    new_body.append(new_class)
    module = ast.Module(new_body, type_ignores=[])
    try:
        return unparse(module)
    except Exception as e:
        print("failed to unparse", e)
        raise e


def check_class_code(context: TransformContext) -> TransformContext:
    # TODO: define the proper checking for this case based on the ideas from UserCode
    # check for no globals
    # check for Policy template -> __init__, apply_output, public_state
    # parse init signature
    # check dangerous libraries, maybe compile_restricted already does that
    try:
        processed_code = process_class_code(
            raw_code=context.output["raw_code"],
            class_name=context.output["unique_name"],
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
        sys.modules["syft"].user = user_module
    user_module = sy.user
    setattr(user_module, unique_name, klass)
    sys.modules["syft"].user = user_module
    return klass


def execute_policy_code(user_policy: UserPolicy):
    stdout_ = sys.stdout
    stderr_ = sys.stderr

    try:
        stdout = StringIO()
        stderr = StringIO()

        sys.stdout = stdout
        sys.stderr = stderr

        class_name = f"{user_policy.unique_name}"
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


def load_policy_code(user_policy: UserPolicy) -> Any:
    try:
        policy_class = execute_policy_code(user_policy)
        return policy_class
    except Exception as e:
        raise Exception(f"Exception loading code. {user_policy}. {e}")


def init_policy(user_policy: UserPolicy, init_args: Dict[str, Any]):
    policy_class = load_policy_code(user_policy)
    policy_object = policy_class()
    policy_object.__user_init__(**init_args)
    return policy_object
