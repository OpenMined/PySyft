# future
from __future__ import annotations

# stdlib
import ast
from collections.abc import Callable
from copy import deepcopy
from enum import Enum
import hashlib
import inspect
from inspect import Parameter
from inspect import Signature
from io import StringIO
import sys
from typing import Any
from typing import ClassVar
from typing import TYPE_CHECKING

# third party
from RestrictedPython import compile_restricted
from pydantic import field_validator
from pydantic import model_validator
import requests

# relative
from ...abstract_server import ServerType
from ...client.api import APIRegistry
from ...client.api import RemoteFunction
from ...client.api import ServerIdentity
from ...serde.recursive_primitives import recursive_serde_register_type
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.datetime import DateTime
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.syft_object_registry import SyftObjectRegistry
from ...types.transforms import TransformContext
from ...types.transforms import generate_id
from ...types.transforms import transform
from ...types.twin_object import TwinObject
from ...types.uid import UID
from ...util.util import is_interpreter_jupyter
from ..action.action_endpoint import CustomEndpointActionObject
from ..action.action_object import ActionObject
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from ..code.code_parse import GlobalsVisitor
from ..code.unparse import unparse
from ..context import AuthedServiceContext
from ..context import ChangeContext
from ..context import ServerServiceContext
from ..dataset.dataset import Asset

# Use this for return type enums:
# class MyEnum(Enum):
#     MEMBER1 = ("Value1", True)
#     MEMBER2 = ("Value2", False)
#     MEMBER3 = ("Value3", True)

#     def __init__(self, value: str, flag: bool):
#         self._value_ = value
#         self.flag = flag

# # Example usage:
# for member in MyEnum:
#     print(f"Name: {member.name}, Value: {member.value}, Flag: {member.flag}")


class InputPolicyValidEnum(Enum):
    VALID = "valid"
    INVALID = "invalid"


class OutputPolicyValidEnum(Enum):
    VALID = "valid"
    INVALID = "invalid"
    NOT_APPROVED = "not_approved"


DEFAULT_USER_POLICY_VERSION = 1

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


def filter_only_uids(results: Any) -> list[UID] | dict[str, UID] | UID:
    # Prevent checking for __len__ on ActionObject (creates an Action)
    if isinstance(results, ActionObject):
        return extract_uid(results)

    if not hasattr(results, "__len__"):
        results = [results]

    if isinstance(results, list):
        output_list = [extract_uid(v) for v in results]
        return output_list
    elif isinstance(results, dict):
        output_dict = {}
        for k, v in results.items():
            output_dict[k] = extract_uid(v)
        return output_dict
    return extract_uid(results)


class Policy(SyftObject):
    # version
    __canonical_name__: str = "Policy"
    __version__ = SYFT_OBJECT_VERSION_1
    has_safe_serde: ClassVar[bool] = True

    id: UID
    init_kwargs: dict[Any, Any] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "init_kwargs" in kwargs:
            init_kwargs = kwargs["init_kwargs"]
            del kwargs["init_kwargs"]
        else:
            init_kwargs = deepcopy(kwargs)
            if "id" in init_kwargs:
                del init_kwargs["id"]
        super().__init__(init_kwargs=init_kwargs, *args, **kwargs)  # noqa: B026

    @property
    def policy_code(self) -> str:
        mro = reversed(type(self).mro())
        op_code = ""
        for klass in mro:
            if "Policy" in klass.__name__:
                op_code += inspect.getsource(klass)
                op_code += "\n"
        return op_code

    def is_valid(self, *args: list, **kwargs: dict) -> bool:  # type: ignore
        return True

    def public_state(self) -> Any:
        raise NotImplementedError


@serializable(canonical_name="UserPolicyStatus", version=1)
class UserPolicyStatus(Enum):
    SUBMITTED = "submitted"
    DENIED = "denied"
    APPROVED = "approved"


def partition_by_server(kwargs: dict[str, Any]) -> dict[ServerIdentity, dict[str, UID]]:
    # relative
    from ...client.api import APIRegistry
    from ...client.api import RemoteFunction
    from ...client.api import ServerIdentity
    from ...types.twin_object import TwinObject
    from ..action.action_object import ActionObject

    # fetches the all the current api's connected
    output_kwargs = {}
    for k, v in kwargs.items():
        uid = v
        if isinstance(v, ActionObject):
            uid = v.id
        if isinstance(v, TwinObject):
            uid = v.id
        if isinstance(v, RemoteFunction):
            uid = v.custom_function_actionobject_id().unwrap()
        if isinstance(v, Asset):
            uid = v.action_id
        if not isinstance(uid, UID):
            raise Exception(f"Input {k} must have a UID not {type(v)}")

        _obj_exists = False
        for identity, api in APIRegistry.__api_registry__.items():
            try:
                if api.services.action.exists(uid):
                    server_identity = ServerIdentity.from_api(api)
                    if server_identity not in output_kwargs:
                        output_kwargs[server_identity] = {k: uid}
                    else:
                        output_kwargs[server_identity].update({k: uid})

                    _obj_exists = True
                    break
            except (requests.exceptions.ConnectionError, SyftException):
                # To handle the cases , where there an old api objects in
                # in APIRegistry
                continue
            except Exception as e:
                print(f"Error in partition_by_server with identity {identity}", e)
                raise e

        if not _obj_exists:
            raise Exception(f"Input data {k}:{uid} does not belong to any Datasite")

    return output_kwargs


@serializable()
class PolicyRule(SyftObject):
    __canonical_name__ = "PolicyRule"
    __version__ = SYFT_OBJECT_VERSION_1

    kw: str
    requires_input: bool = True

    def is_met(
        self, context: AuthedServiceContext, action_object: ActionObject
    ) -> bool:
        return False


@serializable()
class CreatePolicyRule(SyftObject):
    __canonical_name__ = "CreatePolicyRule"
    __version__ = SYFT_OBJECT_VERSION_1

    val: Any


@serializable()
class CreatePolicyRuleConstant(CreatePolicyRule):
    __canonical_name__ = "CreatePolicyRuleConstant"
    __version__ = SYFT_OBJECT_VERSION_1

    val: Any
    klass: None | type = None

    @model_validator(mode="before")
    @classmethod
    def set_klass(cls, data: Any) -> Any:
        val = data["val"]
        if isinstance(val, RemoteFunction):
            klass = CustomEndpointActionObject
        else:
            klass = type(val)
        data["klass"] = klass
        return data

    @field_validator("val", mode="after")
    @classmethod
    def idify_endpoints(cls, value: str) -> str:
        if isinstance(value, RemoteFunction):
            return value.custom_function_actionobject_id().unwrap()
        return value

    def to_policy_rule(self, kw: Any) -> PolicyRule:
        return Constant(kw=kw, val=self.val, klass=self.klass)


@serializable()
class Matches(PolicyRule):
    __canonical_name__ = "Matches"
    __version__ = SYFT_OBJECT_VERSION_1

    val: UID

    def is_met(
        self, context: AuthedServiceContext, action_object: ActionObject
    ) -> bool:
        return action_object.id == self.val


@serializable()
class Constant(PolicyRule):
    __canonical_name__ = "PreFill"
    __version__ = SYFT_OBJECT_VERSION_1

    val: Any
    klass: type
    requires_input: bool = False

    @property
    def value(self) -> Any:
        return self.val

    def is_met(self, context: AuthedServiceContext, *args: Any, **kwargs: Any) -> bool:
        return True

    @as_result(SyftException)
    def transform_kwarg(self, context: AuthedServiceContext, val: Any) -> Any:
        if isinstance(self.val, UID):
            if issubclass(self.klass, CustomEndpointActionObject):
                obj = context.server.services.action.get(
                    context.as_root_context(), self.val
                )
                return obj.syft_action_data
        return self.val

    def _get_dict_for_user_code_repr(self) -> dict[str, Any]:
        return self._coll_repr_()

    def _coll_repr_(self) -> dict[str, Any]:
        return {
            "klass": self.klass.__qualname__,
            "val": str(self.val),
        }


@serializable()
class UserOwned(PolicyRule):
    __canonical_name__ = "UserOwned"
    __version__ = SYFT_OBJECT_VERSION_1

    # str, float, int, bool, dict, list, set, tuple

    type: (
        type[str]
        | type[float]
        | type[int]
        | type[bool]
        | type[dict]
        | type[list]
        | type[set]
        | type[tuple]
        | None
    )

    def is_owned(
        self, context: AuthedServiceContext, action_object: ActionObject
    ) -> bool:
        action_store = context.server.services.action.stash
        return action_store.has_permission(
            ActionObjectPermission(
                action_object.id, ActionPermission.OWNER, context.credentials
            )
        )

    def is_met(
        self, context: AuthedServiceContext, action_object: ActionObject
    ) -> bool:
        return type(action_object.syft_action_data) == self.type and self.is_owned(
            context, action_object
        )


def user_code_arg2id(arg: Any) -> UID:
    if isinstance(arg, ActionObject):
        uid = arg.id
    elif isinstance(arg, TwinObject):
        uid = arg.id
    elif isinstance(arg, Asset):
        uid = arg.action_id
    elif isinstance(arg, RemoteFunction):
        # TODO: Beach Fix
        # why do we need another call to the server to get the UID?
        uid = arg.custom_function_actionobject_id().unwrap()
    else:
        uid = arg
    return uid


def retrieve_item_from_db(id: UID, context: AuthedServiceContext) -> ActionObject:
    # relative
    from ...service.action.action_object import TwinMode

    root_context = AuthedServiceContext(
        server=context.server, credentials=context.server.verify_key
    )
    return context.server.services.action._get(
        context=root_context,
        uid=id,
        twin_mode=TwinMode.NONE,
        has_permission=True,
    ).unwrap()


class InputPolicy(Policy):
    __canonical_name__ = "InputPolicy"
    __version__ = SYFT_OBJECT_VERSION_1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "init_kwargs" in kwargs:
            init_kwargs = kwargs["init_kwargs"]
            del kwargs["init_kwargs"]
        else:
            # TODO: remove this tech debt, dont remove the id mapping functionality
            init_kwargs = partition_by_server(kwargs)
        super().__init__(*args, init_kwargs=init_kwargs, **kwargs)

    def is_valid(  # type: ignore
        self,
        context: AuthedServiceContext,
        usr_input_kwargs: dict,
    ) -> bool:
        raise NotImplementedError

    def filter_kwargs(
        self,
        kwargs: dict[Any, Any],
        context: AuthedServiceContext,
    ) -> dict[Any, Any]:
        raise NotImplementedError

    @property
    def inputs(self) -> dict[ServerIdentity, Any]:
        return self.init_kwargs

    def _inputs_for_context(self, context: ChangeContext) -> dict:
        user_server_view = ServerIdentity.from_change_context(context)
        inputs = self.inputs[user_server_view]
        root_context = AuthedServiceContext(
            server=context.server, credentials=context.approving_user_credentials
        ).as_root_context()

        for var_name, uid in inputs.items():
            action_object_value = context.server.services.action.get(
                uid=uid, context=root_context
            ).unwrap()
            # resolve syft action data from blob store
            if isinstance(action_object_value, TwinObject):
                action_object_value.private_obj.syft_action_data  # noqa: B018
                action_object_value.mock_obj.syft_action_data  # noqa: B018
            elif isinstance(action_object_value, ActionObject):
                action_object_value.syft_action_data  # noqa: B018
            inputs[var_name] = action_object_value
        return inputs


@serializable()
class MixedInputPolicy(InputPolicy):
    # version
    __canonical_name__ = "MixedInputPolicy"
    __version__ = SYFT_OBJECT_VERSION_1

    kwarg_rules: dict[ServerIdentity, dict[str, PolicyRule]]

    def __init__(
        self, init_kwargs: Any = None, client: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        if init_kwargs is not None:
            kwarg_rules = init_kwargs
            kwargs = {}
        else:
            server_identity = self.find_server_identity(kwargs, client)
            kwarg_rules_current_server = {}
            for kw, arg in kwargs.items():
                if isinstance(
                    arg, UID | Asset | ActionObject | TwinObject | RemoteFunction
                ):
                    kwarg_rules_current_server[kw] = Matches(
                        kw=kw, val=user_code_arg2id(arg)
                    )
                elif arg in [str, float, int, bool, dict, list, set, tuple]:  # type: ignore[unreachable]
                    kwarg_rules_current_server[kw] = UserOwned(kw=kw, type=arg)
                elif isinstance(arg, CreatePolicyRule):
                    kwarg_rules_current_server[kw] = arg.to_policy_rule(kw)
                else:
                    raise ValueError("Incorrect argument")
            kwarg_rules = {server_identity: kwarg_rules_current_server}

        super().__init__(
            *args, kwarg_rules=kwarg_rules, init_kwargs=kwarg_rules, **kwargs
        )

    @as_result(SyftException)
    def transform_kwargs(
        self, context: AuthedServiceContext, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        for _, rules in self.kwarg_rules.items():
            for kw, rule in rules.items():
                if hasattr(rule, "transform_kwarg"):
                    kwargs[kw] = rule.transform_kwarg(
                        context, kwargs.get(kw, None)
                    ).unwrap()
        return kwargs

    def find_server_identity(
        self, kwargs: dict[str, Any], client: Any = None
    ) -> ServerIdentity:
        if client is not None:
            return ServerIdentity.from_api(client.api)

        apis = APIRegistry.get_all_api()
        matches = set()
        has_ids = False
        for val in kwargs.values():
            # we mostly get the UID here because we don't want to store all those
            # other objects, so we need to create a global UID obj lookup service
            if isinstance(
                val, UID | Asset | ActionObject | TwinObject | RemoteFunction
            ):
                has_ids = True
                id = user_code_arg2id(val)
                for api in apis:
                    # TODO: Beach Fix
                    # here be dragons, we need to refactor this since the existance
                    # depends on the type and service
                    # also the whole ServerIdentity needs to be removed
                    check_endpoints = [
                        api.services.action.exists,
                        api.services.api.exists,
                    ]
                    for check_endpoint in check_endpoints:
                        result = check_endpoint(id)
                        if result:
                            break  # stop looking
                    if result:
                        server_identity = ServerIdentity.from_api(api)
                        matches.add(server_identity)

        if len(matches) == 0:
            if not has_ids:
                if len(apis) == 1:
                    return ServerIdentity.from_api(api)
                else:
                    raise ValueError(
                        "Multiple Server Identities, please only login to one client (for this policy) and try again"
                    )
            else:
                raise ValueError("No Server Identities")
        if len(matches) > 1:
            # TODO: Beach Fix
            raise ValueError("Multiple Server Identities")
            # we need to fix this as its possible we could
            # grab the wrong API and call a different user context in jupyter testing
            pass  # just grab the first one
        return matches.pop()

    def filter_kwargs(  # type: ignore[override]
        self,
        kwargs: dict[str, UID],
        context: AuthedServiceContext,
    ) -> dict[Any, Any]:
        try:
            res = {}
            for _, rules in self.kwarg_rules.items():
                for kw, rule in rules.items():
                    if rule.requires_input:
                        passed_id = kwargs[kw]
                        actionobject: ActionObject = retrieve_item_from_db(
                            passed_id, context
                        )
                        rule_check_args = (actionobject,)
                    else:
                        rule_check_args = ()  # type: ignore
                        # TODO
                        actionobject = rule.value
                    if not rule.is_met(context, *rule_check_args):
                        raise SyftException(public_message=f"{rule} is not met")
                    else:
                        res[kw] = actionobject
        except Exception as e:
            raise SyftException.from_exception(
                e, public_message="failed to filter kwargs"
            )
        return res

    def is_valid(  # type: ignore[override]
        self,
        context: AuthedServiceContext,
        usr_input_kwargs: dict,
    ) -> bool:
        filtered_input_kwargs = self.filter_kwargs(
            kwargs=usr_input_kwargs,
            context=context,
        )
        expected_input_kwargs = set()

        for _inp_kwargs in self.inputs.values():
            for k in _inp_kwargs.keys():
                if k not in usr_input_kwargs and k not in filtered_input_kwargs:
                    raise SyftException(
                        public_message=f"Function missing required keyword argument: '{k}'"
                    )
            expected_input_kwargs.update(_inp_kwargs.keys())

        permitted_input_kwargs = list(filtered_input_kwargs.keys())
        not_approved_kwargs = set(expected_input_kwargs) - set(permitted_input_kwargs)

        if len(not_approved_kwargs) > 0:
            raise SyftException(
                public_message=f"Input arguments: {not_approved_kwargs} to the function are not approved yet."
            )

        return True


@as_result(SyftException, NotFoundException, StashException)
def retrieve_from_db(
    allowed_inputs: dict[str, UID], context: AuthedServiceContext
) -> dict[str, Any]:
    # relative
    from ...service.action.action_object import TwinMode

    if TYPE_CHECKING:
        # relative
        pass

    code_inputs = {}

    # When we are retrieving the code from the database, we need to use the server's
    # verify key as the credentials. This is because when we approve the code, we
    # we allow the private data to be used only for this specific code.
    # but we are not modifying the permissions of the private data
    root_context = AuthedServiceContext(
        server=context.server, credentials=context.server.verify_key
    )

    if context.server.server_type != ServerType.DATASITE:
        raise SyftException(
            public_message=f"Invalid server type for code submission: {context.server.server_type}"
        )

    for var_name, arg_id in allowed_inputs.items():
        code_inputs[var_name] = context.server.services.action._get(
            context=root_context,
            uid=arg_id,
            twin_mode=TwinMode.NONE,
            has_permission=True,
        ).unwrap()

    return code_inputs


@as_result(SyftException)
def allowed_ids_only(
    allowed_inputs: dict[ServerIdentity, Any],
    kwargs: dict[str, Any],
    context: AuthedServiceContext,
) -> dict[ServerIdentity, UID]:
    if context.server.server_type != ServerType.DATASITE:
        raise SyftException(
            public_message=f"Invalid server type for code submission: {context.server.server_type}"
        )

    allowed_inputs_for_server = None
    for identity, inputs in allowed_inputs.items():
        if identity.server_id == context.server.id:
            allowed_inputs_for_server = inputs
            break
    if allowed_inputs_for_server is None:
        allowed_inputs_for_server = {}

    filtered_kwargs = {}
    for key in allowed_inputs_for_server.keys():
        if key in kwargs:
            value = kwargs[key]
            uid = value

            if not isinstance(uid, UID):
                uid = getattr(value, "id", None)

            if uid != allowed_inputs_for_server[key]:
                raise SyftException(
                    public_message=f"Input with uid: {uid} for `{key}` not in allowed inputs: {allowed_inputs}"
                )

            filtered_kwargs[key] = value

    return filtered_kwargs


@serializable()
class ExactMatch(InputPolicy):
    # version
    __canonical_name__ = "ExactMatch"
    __version__ = SYFT_OBJECT_VERSION_1

    # TODO: Improve exception handling here
    def filter_kwargs(  # type: ignore
        self,
        kwargs: dict[Any, Any],
        context: AuthedServiceContext,
    ) -> dict[Any, Any]:
        allowed_inputs = allowed_ids_only(
            allowed_inputs=self.inputs, kwargs=kwargs, context=context
        ).unwrap()

        return retrieve_from_db(
            allowed_inputs=allowed_inputs,
            context=context,
        ).unwrap()

    def is_valid(  # type: ignore
        self,
        context: AuthedServiceContext,
        usr_input_kwargs: dict,
    ) -> bool:
        filtered_input_kwargs = self.filter_kwargs(
            kwargs=usr_input_kwargs,
            context=context,
        )

        expected_input_kwargs = set()
        for _inp_kwargs in self.inputs.values():
            for k in _inp_kwargs.keys():
                if k not in usr_input_kwargs:
                    raise SyftException(
                        public_message=f"Function missing required keyword argument: '{k}'"
                    )
            expected_input_kwargs.update(_inp_kwargs.keys())

        permitted_input_kwargs = list(filtered_input_kwargs.keys())

        not_approved_kwargs = set(expected_input_kwargs) - set(permitted_input_kwargs)
        if len(not_approved_kwargs) > 0:
            raise SyftException(
                public_message=f"Function arguments: {not_approved_kwargs} are not approved yet."
            )

        return True


@serializable()
class OutputHistory(SyftObject):
    # version
    __canonical_name__ = "OutputHistory"
    __version__ = SYFT_OBJECT_VERSION_1

    output_time: DateTime
    outputs: list[UID] | dict[str, UID] | None = None
    executing_user_verify_key: SyftVerifyKey


class OutputPolicy(Policy):
    # version
    __canonical_name__ = "OutputPolicy"
    __version__ = SYFT_OBJECT_VERSION_1

    output_kwargs: list[str] = []
    server_uid: UID | None = None
    output_readers: list[SyftVerifyKey] = []

    def apply_to_output(
        self,
        context: ServerServiceContext,
        outputs: Any,
        update_policy: bool = True,
    ) -> Any:
        # output_uids: Union[Dict[str, Any], list] = filter_only_uids(outputs)
        # if isinstance(output_uids, UID):
        #     output_uids = [output_uids]
        # history = OutputHistory(
        #     output_time=DateTime.now(),
        #     outputs=output_uids,
        #     executing_user_verify_key=context.credentials,
        # )
        # self.output_history.append(history)

        return outputs

    def is_valid(self, context: AuthedServiceContext | None) -> bool:  # type: ignore
        raise NotImplementedError()


@serializable()
class OutputPolicyExecuteCount(OutputPolicy):
    __canonical_name__ = "OutputPolicyExecuteCount"
    __version__ = SYFT_OBJECT_VERSION_1

    limit: int

    # def is_valid(self, context: AuthedServiceContext) -> bool:
    #     return self.count().unwrap() < self.limit

    # @as_result(SyftException)
    # def count(self) -> int:
    #     api = self.get_api()
    #     output_history = api.services.output.get_by_output_policy_id(self.id)
    #     return len(output_history)

    def count(self, context: AuthedServiceContext | None = None) -> int:
        # client side
        if context is None:
            output_service = self.get_api().services.output
            output_history = output_service.get_by_output_policy_id(self.id)
        else:
            # server side
            output_history = context.server.services.output.get_by_output_policy_id(
                context, self.id
            )  # raises

        return len(output_history)

    def is_valid(self, context: AuthedServiceContext | None = None) -> bool:  # type: ignore
        return self.count(context) < self.limit

    def public_state(self) -> dict[str, int]:
        # TODO: this count is not great, fix it.
        return {"limit": self.limit, "count": self.count().unwrap()}


@serializable()
class OutputPolicyExecuteOnce(OutputPolicyExecuteCount):
    __canonical_name__ = "OutputPolicyExecuteOnce"
    __version__ = SYFT_OBJECT_VERSION_1

    limit: int = 1


SingleExecutionExactOutput = OutputPolicyExecuteOnce


@serializable(canonical_name="CustomPolicy", version=1)
class CustomPolicy(type):
    # capture the init_kwargs transparently
    def __call__(cls, *args: Any, **kwargs: Any) -> None:
        obj = super().__call__(*args, **kwargs)
        obj.init_kwargs = kwargs
        return obj


recursive_serde_register_type(CustomPolicy, canonical_name="CustomPolicy", version=1)


@serializable(canonical_name="CustomOutputPolicy", version=1)
class CustomOutputPolicy(metaclass=CustomPolicy):
    def apply_to_output(
        self,
        context: ServerServiceContext,
        outputs: Any,
        update_policy: bool = True,
    ) -> Any | None:
        return outputs


class UserOutputPolicy(OutputPolicy):
    __canonical_name__ = "UserOutputPolicy"

    # Do not validate private attributes of user-defined policies, User annotations can
    # contain any type and throw a NameError when resolving.
    __validate_private_attrs__ = False
    pass


class UserInputPolicy(InputPolicy):
    __canonical_name__ = "UserInputPolicy"
    __validate_private_attrs__ = False
    pass


@serializable()
class EmpyInputPolicy(InputPolicy):
    __canonical_name__ = "EmptyInputPolicy"
    pass


class CustomInputPolicy(metaclass=CustomPolicy):
    pass


@serializable()
class UserPolicy(Policy):
    __canonical_name__: str = "UserPolicy"
    __version__ = SYFT_OBJECT_VERSION_1
    has_safe_serde: ClassVar[bool] = False

    id: UID
    server_uid: UID | None = None
    user_verify_key: SyftVerifyKey
    raw_code: str
    parsed_code: str
    signature: inspect.Signature
    class_name: str
    unique_name: str
    code_hash: str
    status: UserPolicyStatus = UserPolicyStatus.SUBMITTED

    # TODO: fix the mypy issue
    @property  # type: ignore
    def byte_code(self) -> PyCodeObject | None:
        return compile_byte_code(self.parsed_code)

    @property
    def policy_code(self) -> str:
        return self.raw_code

    def apply_to_output(
        self,
        context: ServerServiceContext,
        outputs: Any,
        update_policy: bool = True,
    ) -> Any | None:
        return outputs


def new_getfile(object: Any) -> Any:  # TODO: fix the mypy issue
    if not inspect.isclass(object):
        return inspect.getfile(object)

    # Lookup by parent module (as in current inspect)
    if hasattr(object, "__module__"):
        object_ = sys.modules.get(object.__module__)
        if object_ is not None and hasattr(object_, "__file__"):
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


def get_code_from_class(policy: type[CustomPolicy]) -> str:
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

    id: UID | None = None  # type: ignore[assignment]
    code: str
    class_name: str
    input_kwargs: list[str]

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
    if context.output is None:
        return context
    code = context.output["code"]
    del context.output["code"]
    context.output["raw_code"] = code
    code_hash = hashlib.sha256(code.encode("utf8")).hexdigest()
    context.output["code_hash"] = code_hash

    return context


def generate_unique_class_name(context: TransformContext) -> TransformContext:
    # TODO: Do we need to check if the initial name contains underscores?
    if context.output is not None:
        code_hash = context.output["code_hash"]
        service_class_name = context.output["class_name"]
        unique_name = f"{service_class_name}_{context.credentials}_{code_hash}"
        context.output["unique_name"] = unique_name
    else:
        raise ValueError(f"{context}'s output is None. No transformation happened")

    return context


def compile_byte_code(parsed_code: str) -> PyCodeObject | None:
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
        raise SyftException(
            public_message="Class code should only contain the Class definition for your policy"
        )
    old_class = tree.body[0]
    if len(old_class.bases) != 1 or old_class.bases[0].attr not in [
        CustomInputPolicy.__name__,
        CustomOutputPolicy.__name__,
    ]:
        raise SyftException(
            public_message=(
                f"Class code should either implement {CustomInputPolicy.__name__}"
                f" or {CustomOutputPolicy.__name__}"
            )
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
        if isinstance(stmt, ast.FunctionDef):
            if stmt.name == "__init__":
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
    new_body.append(
        ast.ImportFrom(
            module="typing",
            names=[
                ast.alias(name=typing_type, asname=typing_type)
                for typing_type in typing_types
            ],
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
    # check for Policy template -> __init__, apply_to_output, public_state
    # parse init signature
    # check dangerous libraries, maybe compile_restricted already does that
    if context.output is None:
        return context

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
    if context.output is not None:
        byte_code = compile_byte_code(context.output["parsed_code"])
        if byte_code is None:
            raise Exception(
                "Unable to compile byte code from parsed code. "
                + context.output["parsed_code"]
            )
    else:
        raise ValueError(f"{context}'s output is None. No transformation happened")

    return context


def add_credentials_for_key(key: str) -> Callable:
    def add_credentials(context: TransformContext) -> TransformContext:
        if context.output is not None:
            context.output[key] = context.credentials

        return context

    return add_credentials


def generate_signature(context: TransformContext) -> TransformContext:
    if context.output is None:
        return context

    params = [
        Parameter(name=k, kind=Parameter.POSITIONAL_OR_KEYWORD)
        for k in context.output["input_kwargs"]
    ]
    sig = Signature(parameters=params)
    context.output["signature"] = sig

    return context


@transform(SubmitUserPolicy, UserPolicy)
def submit_policy_code_to_user_code() -> list[Callable]:
    return [
        generate_id,
        hash_code,
        generate_unique_class_name,
        generate_signature,
        check_class_code,
        # compile_code, # don't compile until approved
        add_credentials_for_key("user_verify_key"),
    ]


def register_policy_class(klass: type, unique_name: str) -> None:
    nonrecursive = False
    _serialize = None
    _deserialize = None
    attributes = list(klass.model_fields.keys())
    exclude_attrs: list = []
    serde_overrides: dict = {}
    hash_exclude_attrs: list = []
    cls = klass
    attribute_types: list = []
    version = 1

    serde_attributes = (
        nonrecursive,
        _serialize,
        _deserialize,
        attributes,
        exclude_attrs,
        serde_overrides,
        hash_exclude_attrs,
        cls,
        attribute_types,
        version,
    )

    SyftObjectRegistry.register_cls(
        canonical_name=unique_name, version=version, serde_attributes=serde_attributes
    )


def execute_policy_code(user_policy: UserPolicy) -> Any:
    stdout_ = sys.stdout
    stderr_ = sys.stderr

    try:
        stdout = StringIO()
        stderr = StringIO()

        sys.stdout = stdout
        sys.stderr = stderr

        class_name = user_policy.unique_name

        try:
            policy_class = SyftObjectRegistry.get_serde_class(
                class_name, version=DEFAULT_USER_POLICY_VERSION
            )
        except Exception:
            exec(user_policy.byte_code)  # nosec
            policy_class = eval(user_policy.unique_name)  # nosec

        policy_class.model_rebuild()
        register_policy_class(policy_class, user_policy.unique_name)

        sys.stdout = stdout_
        sys.stderr = stderr_

        return policy_class

    except Exception as e:
        print(
            f"execute_byte_code failed because of {e}, with the following code\n\n{user_policy.parsed_code}",
            file=stderr_,
        )

    finally:
        sys.stdout = stdout_
        sys.stderr = stderr_


def load_policy_code(user_policy: UserPolicy) -> Any:
    try:
        policy_class = execute_policy_code(user_policy)
        return policy_class
    except SyftException as exc:
        raise SyftException.from_exception(
            exc, public_message=f"Exception loading code. {user_policy}."
        )


def init_policy(user_policy: UserPolicy, init_args: dict[str, Any]) -> Any:
    policy_class = load_policy_code(user_policy)
    policy_object = policy_class()

    # Unwrapp {ServerIdentity : {x: y}} -> {x: y}
    # Tech debt : For input policies, we required to have ServerIdentity args beforehand,
    # therefore at this stage we had to return back to the normal args.
    # Maybe there's better way to do it.
    if len(init_args) and isinstance(list(init_args.keys())[0], ServerIdentity):
        unwrapped_init_kwargs = init_args
        if len(init_args) > 1:
            raise SyftException(
                public_message="You shoudn't have more than one Server Identity."
            )
        # Otherwise, unwrap it
        init_args = init_args[list(init_args.keys())[0]]

    init_args = {k: v for k, v in init_args.items() if k != "id"}

    # For input policies, this initializer wouldn't work properly:
    # 1 - Passing {ServerIdentity: {kwargs:UIDs}} as keyword args doesn't work since keys must be strings
    # 2 - Passing {kwargs: UIDs} in this initializer would not trigger the partition servers from the
    # InputPolicy initializer.
    # The cleanest way to solve it is by checking if it's an Input Policy, and then, setting it manually.
    policy_object.__user_init__(**init_args)
    if isinstance(policy_object, InputPolicy):
        policy_object.init_kwargs = unwrapped_init_kwargs
    return policy_object
