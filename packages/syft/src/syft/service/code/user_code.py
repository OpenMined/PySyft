# future
from __future__ import annotations

# stdlib
import ast
from collections.abc import Callable
from copy import deepcopy
import datetime
from enum import Enum
import hashlib
import inspect
from io import StringIO
import json
import keyword
import logging
import random
import re
import sys
from textwrap import dedent
from threading import Thread
import time
import traceback
from typing import Any
from typing import ClassVar
from typing import TYPE_CHECKING
from typing import cast
from typing import final

# third party
from IPython.display import HTML
from IPython.display import Markdown
from IPython.display import display
from pydantic import ValidationError
from pydantic import field_validator
from typing_extensions import Self

# relative
from ...abstract_server import ServerSideType
from ...abstract_server import ServerType
from ...client.api import APIRegistry
from ...client.api import ServerIdentity
from ...client.api import generate_remote_function
from ...serde.deserialize import _deserialize
from ...serde.serializable import serializable
from ...serde.serialize import _serialize
from ...serde.signature import signature_remove_context
from ...serde.signature import signature_remove_self
from ...server.credentials import SyftVerifyKey
from ...store.document_store import PartitionKey
from ...store.linked_obj import LinkedObject
from ...types.datetime import DateTime
from ...types.dicttuple import DictTuple
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.syft_object import PartialSyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject
from ...types.syncable_object import SyncableSyftObject
from ...types.transforms import TransformContext
from ...types.transforms import add_server_uid_for_key
from ...types.transforms import generate_id
from ...types.transforms import transform
from ...types.uid import UID
from ...util.decorators import deprecated
from ...util.markdown import CodeMarkdown
from ...util.markdown import as_markdown_code
from ...util.notebook_ui.styles import FONT_CSS
from ...util.util import prompt_warning_message
from ..action.action_endpoint import CustomEndpointActionObject
from ..action.action_object import Action
from ..action.action_object import ActionObject
from ..context import AuthedServiceContext
from ..dataset.dataset import Asset
from ..job.job_stash import Job
from ..output.output_service import ExecutionOutput
from ..output.output_service import OutputService
from ..policy.policy import Constant
from ..policy.policy import CustomInputPolicy
from ..policy.policy import CustomOutputPolicy
from ..policy.policy import EmpyInputPolicy
from ..policy.policy import ExactMatch
from ..policy.policy import InputPolicy
from ..policy.policy import OutputPolicy
from ..policy.policy import SingleExecutionExactOutput
from ..policy.policy import SubmitUserPolicy
from ..policy.policy import UserPolicy
from ..policy.policy import filter_only_uids
from ..policy.policy import init_policy
from ..policy.policy import load_policy_code
from ..policy.policy import partition_by_server
from ..policy.policy_service import PolicyService
from ..response import SyftError
from ..response import SyftInfo
from ..response import SyftSuccess
from ..response import SyftWarning
from ..service import ServiceConfigRegistry
from ..user.user import UserView
from ..user.user_roles import ServiceRole
from .code_parse import LaunchJobVisitor
from .unparse import unparse
from .utils import check_for_global_vars
from .utils import parse_code
from .utils import submit_subjobs_code

logger = logging.getLogger(name=__name__)

if TYPE_CHECKING:
    # relative
    from ...service.sync.diff_state import AttrDiff

UserVerifyKeyPartitionKey = PartitionKey(key="user_verify_key", type_=SyftVerifyKey)
CodeHashPartitionKey = PartitionKey(key="code_hash", type_=str)
ServiceFuncNamePartitionKey = PartitionKey(key="service_func_name", type_=str)
SubmitTimePartitionKey = PartitionKey(key="submit_time", type_=DateTime)

PyCodeObject = Any


def compile_byte_code(parsed_code: str) -> PyCodeObject | None:
    try:
        return compile(parsed_code, "<string>", "exec")
    except Exception as e:
        print("WARNING: to compile byte code", e)
    return None


@serializable(canonical_name="UserCodeStatus", version=1)
class UserCodeStatus(Enum):
    PENDING = "pending"
    DENIED = "denied"
    APPROVED = "approved"

    def __hash__(self) -> int:
        return hash(self.value)


@serializable()
class UserCodeStatusCollection(SyncableSyftObject):
    __canonical_name__ = "UserCodeStatusCollection"
    __version__ = SYFT_OBJECT_VERSION_1

    __repr_attrs__ = ["approved", "status_dict"]
    status_dict: dict[ServerIdentity, tuple[UserCodeStatus, str]] = {}
    user_code_link: LinkedObject

    def syft_get_diffs(self, ext_obj: Any) -> list[AttrDiff]:
        # relative
        from ...service.sync.diff_state import AttrDiff

        diff_attrs = []
        status = list(self.status_dict.values())[0]
        ext_status = list(ext_obj.status_dict.values())[0]

        if status != ext_status:
            diff_attr = AttrDiff(
                attr_name="status_dict",
                low_attr=status,
                high_attr=ext_status,
            )
            diff_attrs.append(diff_attr)
        return diff_attrs

    def __repr__(self) -> str:
        return str(self.status_dict)

    def _repr_html_(self) -> str:
        string = """
                <div class='syft-user_code'>
                    <h3 style="line-height: 25%; margin-top: 25px;">User Code Status</h3>
                    <p style="margin-left: 3px;">
            """
        for server_identity, (status, reason) in self.status_dict.items():
            server_name_str = f"{server_identity.server_name}"
            uid_str = f"{server_identity.server_id}"
            status_str = f"{status.value}"
            string += f"""
                    &#x2022; <strong>UID: </strong>{uid_str}&nbsp;
                    <strong>Server name: </strong>{server_name_str}&nbsp;
                    <strong>Status: </strong>{status_str};
                    <strong>Reason: </strong>{reason}
                    <br>
                """
        string += "</p></div>"
        return string

    def __repr_syft_nested__(self) -> str:
        string = ""
        for server_identity, (status, reason) in self.status_dict.items():
            string += f"{server_identity.server_name}: {status}, {reason}<br>"
        return string

    def get_status_message(self) -> str:
        if self.approved:
            return f"{type(self)} approved"
        denial_string = ""
        string = ""
        for server_identity, (status, reason) in self.status_dict.items():
            denial_string += f"Code status on server '{server_identity.server_name}' is '{status}'. Reason: {reason}"
            if not reason.endswith("."):
                denial_string += "."
            string += (
                f"Code status on server '{server_identity.server_name}' is '{status}'."
            )
        if self.denied:
            return f"{type(self)} Your code cannot be run: {denial_string}"
        else:
            return f"{type(self)} Your code is waiting for approval. {string}"

    @property
    def approved(self) -> bool:
        return all(x == UserCodeStatus.APPROVED for x, _ in self.status_dict.values())

    @property
    def denied(self) -> bool:
        for status, _ in self.status_dict.values():
            if status == UserCodeStatus.DENIED:
                return True
        return False

    def for_user_context(self, context: AuthedServiceContext) -> UserCodeStatus:
        if context.server.server_type == ServerType.ENCLAVE:
            keys = {status for status, _ in self.status_dict.values()}
            if len(keys) == 1 and UserCodeStatus.APPROVED in keys:
                return UserCodeStatus.APPROVED
            elif UserCodeStatus.PENDING in keys and UserCodeStatus.DENIED not in keys:
                return UserCodeStatus.PENDING
            elif UserCodeStatus.DENIED in keys:
                return UserCodeStatus.DENIED
            else:
                raise Exception(f"Invalid types in {keys} for Code Submission")

        elif context.server.server_type == ServerType.DATASITE:
            server_identity = ServerIdentity(
                server_name=context.server.name,
                server_id=context.server.id,
                verify_key=context.server.signing_key.verify_key,
            )
            if server_identity in self.status_dict:
                return self.status_dict[server_identity][0]
            else:
                raise Exception(
                    f"Code Object does not contain {context.server.name} Datasite's data"
                )
        else:
            raise Exception(
                f"Invalid Server Type for Code Submission:{context.server.server_type}"
            )

    @as_result(SyftException)
    def mutate(
        self,
        value: tuple[UserCodeStatus, str],
        server_name: str,
        server_id: UID,
        verify_key: SyftVerifyKey,
    ) -> Self:
        server_identity = ServerIdentity(
            server_name=server_name, server_id=server_id, verify_key=verify_key
        )
        status_dict = self.status_dict
        if server_identity in status_dict:
            status_dict[server_identity] = value
            self.status_dict = status_dict
            return self
        else:
            raise SyftException(
                public_message="Cannot Modify Status as the Datasite's data is not included in the request"
            )

    def get_sync_dependencies(self, context: AuthedServiceContext) -> list[UID]:
        return [self.user_code_link.object_uid]


@serializable()
class UserCode(SyncableSyftObject):
    # version
    __canonical_name__ = "UserCode"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    server_uid: UID | None = None
    user_verify_key: SyftVerifyKey
    raw_code: str
    input_policy_type: type[InputPolicy] | UserPolicy
    input_policy_init_kwargs: dict[Any, Any] | None = None
    input_policy_state: bytes = b""
    output_policy_type: type[OutputPolicy] | UserPolicy
    output_policy_init_kwargs: dict[Any, Any] | None = None
    output_policy_state: bytes = b""
    parsed_code: str
    service_func_name: str
    unique_func_name: str
    user_unique_func_name: str
    code_hash: str
    signature: inspect.Signature
    status_link: LinkedObject | None = None
    input_kwargs: list[str]
    submit_time: DateTime | None = None
    # tracks if the code calls datasite.something, variable is set during parsing
    uses_datasite: bool = False

    nested_codes: dict[str, tuple[LinkedObject, dict]] | None = {}
    worker_pool_name: str | None = None
    origin_server_side_type: ServerSideType
    l0_deny_reason: str | None = None
    _has_output_read_permissions_cache: bool | None = None

    __table_coll_widths__ = [
        "min-content",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
    ]

    __attr_searchable__: ClassVar[list[str]] = [
        "user_verify_key",
        "service_func_name",
        "code_hash",
    ]
    __attr_unique__: ClassVar[list[str]] = []
    __repr_attrs__: ClassVar[list[str]] = [
        "service_func_name",
        "input_owners",
        "code_status",
        "worker_pool_name",
        "l0_deny_reason",
        "raw_code",
    ]

    __exclude_sync_diff_attrs__: ClassVar[list[str]] = [
        "server_uid",
        "code_status",
        "input_policy_type",
        "input_policy_init_kwargs",
        "input_policy_state",
        "output_policy_type",
        "output_policy_init_kwargs",
        "output_policy_state",
    ]

    @field_validator("service_func_name", mode="after")
    @classmethod
    def service_func_name_is_valid(cls, value: str) -> str:
        _ = is_valid_usercode_name(
            value
        ).unwrap()  # this will throw an error if not valid
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        # Get the attribute from the class, it might be a descriptor or None
        attr = getattr(type(self), key, None)
        # Check if the attribute is a data descriptor
        if inspect.isdatadescriptor(attr):
            if hasattr(attr, "fset"):
                attr.fset(self, value)
            else:
                raise AttributeError(f"The attribute {key} is not settable.")
        else:
            return super().__setattr__(key, value)

    def _coll_repr_(self) -> dict[str, Any]:
        status = [status for status, _ in self.status.status_dict.values()][0].value
        if status == UserCodeStatus.PENDING.value:
            badge_color = "badge-purple"
        elif status == UserCodeStatus.APPROVED.value:
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
            "Submit time": str(self.submit_time),
        }

    @property
    def is_l0_deployment(self) -> bool:
        return self.origin_server_side_type == ServerSideType.LOW_SIDE

    @property
    def is_l2_deployment(self) -> bool:
        return self.origin_server_side_type == ServerSideType.HIGH_SIDE

    @property
    def user(self) -> UserView:
        api = self.get_api()
        return api.services.user.get_by_verify_key(self.user_verify_key)

    def _compute_status_l0(
        self, context: AuthedServiceContext | None = None
    ) -> UserCodeStatusCollection:
        if context is None:
            # Clientside
            api = self._get_api()
            server_identity = ServerIdentity.from_api(api)

            if self._has_output_read_permissions_cache is None:
                is_approved = api.output.has_output_read_permissions(
                    self.id, self.user_verify_key
                )
                self._has_output_read_permissions_cache = is_approved
            else:
                is_approved = self._has_output_read_permissions_cache
        else:
            # Serverside
            server_identity = ServerIdentity.from_server(context.server)
            output_service = context.server.get_service("outputservice")
            is_approved = output_service.has_output_read_permissions(
                context, self.id, self.user_verify_key
            )
        is_denied = self.l0_deny_reason is not None

        if is_denied:
            if is_approved:
                prompt_warning_message(
                    "This request already has results published to the data scientist. "
                    "They will still be able to access those results."
                )
            message = self.l0_deny_reason
            status = (UserCodeStatus.DENIED, message)
        elif is_approved:
            status = (UserCodeStatus.APPROVED, "")
        else:
            status = (UserCodeStatus.PENDING, "")
        status_dict = {server_identity: status}

        return UserCodeStatusCollection(
            status_dict=status_dict,
            user_code_link=LinkedObject.from_obj(self),
        )

    @property
    def status(self) -> UserCodeStatusCollection:
        # Clientside only

        if self.is_l0_deployment:
            if self.status_link is not None:
                raise SyftException(
                    public_message="Encountered a low side UserCode object with a status_link."
                )
            return self._compute_status_l0()

        if self.status_link is None:
            raise SyftException(
                public_message="This UserCode does not have a status. Please contact the Admin."
            )
        res = self.status_link.resolve
        return res

    @as_result(SyftException)
    def get_status(self, context: AuthedServiceContext) -> UserCodeStatusCollection:
        if self.is_l0_deployment:
            if self.status_link is not None:
                raise SyftException(
                    public_message="Encountered a low side UserCode object with a status_link."
                )
            return self._compute_status_l0(context)

        if self.status_link is None:
            raise SyftException(
                public_message="This UserCode does not have a status. Please contact the Admin."
            )

        return self.status_link.resolve_with_context(context).unwrap()

    @as_result(SyftException)
    def is_status_approved(self, context: AuthedServiceContext) -> bool:
        status = self.get_status(context).unwrap()
        return status.approved

    @property
    def input_owners(self) -> list[str] | None:
        if self.input_policy_init_kwargs is not None:
            return [str(x.server_name) for x in self.input_policy_init_kwargs.keys()]
        return None

    @property
    def input_owner_verify_keys(self) -> list[SyftVerifyKey] | None:
        if self.input_policy_init_kwargs is not None:
            return [x.verify_key for x in self.input_policy_init_kwargs.keys()]
        return None

    @property
    def output_reader_names(self) -> list[SyftVerifyKey] | None:
        if (
            self.input_policy_init_kwargs is not None
            and self.output_policy_init_kwargs is not None
        ):
            keys = self.output_policy_init_kwargs.get("output_readers", [])
            inpkey2name = {
                x.verify_key: x.server_name for x in self.input_policy_init_kwargs
            }
            return [inpkey2name[k] for k in keys if k in inpkey2name]
        return None

    @property
    def output_readers(self) -> list[SyftVerifyKey] | None:
        if self.output_policy_init_kwargs is not None:
            return self.output_policy_init_kwargs.get("output_readers", [])
        return None

    @property
    def code_status(self) -> list:
        status_list = []
        for server_view, (status, _) in self.status.status_dict.items():
            status_list.append(
                f"Server: {server_view.server_name}, Status: {status.value}",
            )
        return status_list

    @property
    def input_policy(self) -> InputPolicy | None:
        if self.status.approved or self.input_policy_type.has_safe_serde:
            return self._get_input_policy()
        return None

    def get_input_policy(self, context: AuthedServiceContext) -> InputPolicy | None:
        status = self.get_status(context).unwrap()
        if status.approved or self.input_policy_type.has_safe_serde:
            return self._get_input_policy()
        return None

    # TODO: Change the return type to follow the enum pattern + input policy
    def _get_input_policy(self) -> InputPolicy | None:
        if len(self.input_policy_state) == 0:
            input_policy = None
            if (
                isinstance(self.input_policy_type, type)
                and issubclass(self.input_policy_type, InputPolicy)
                and self.input_policy_init_kwargs is not None
            ):
                # TODO: Tech Debt here
                server_view_workaround = False
                for k in self.input_policy_init_kwargs.keys():
                    if isinstance(k, ServerIdentity):
                        server_view_workaround = True

                if server_view_workaround:
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

    @as_result(SyftException)
    def is_output_policy_approved(self, context: AuthedServiceContext) -> bool:
        status = self.get_status(context).unwrap()
        return status.approved

    @input_policy.setter  # type: ignore
    def input_policy(self, value: Any) -> None:  # type: ignore
        if isinstance(value, InputPolicy):
            self.input_policy_state = _serialize(value, to_bytes=True)
        elif (isinstance(value, bytes) and len(value) == 0) or value is None:
            self.input_policy_state = b""
        else:
            raise Exception(f"You can't set {type(value)} as input_policy_state")

    def get_output_policy(self, context: AuthedServiceContext) -> OutputPolicy | None:
        status = self.get_status(context).unwrap()
        if status.approved or self.output_policy_type.has_safe_serde:
            return self._get_output_policy()
        return None

    @property
    def output_policy(self) -> OutputPolicy | None:  # type: ignore
        if self.status.approved or self.output_policy_type.has_safe_serde:
            return self._get_output_policy()
        return None

    # FIX: change return type like _get_input_policy
    def _get_output_policy(self) -> OutputPolicy | None:
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
                output_policy.syft_server_location = self.syft_server_location
                output_policy.syft_client_verify_key = self.syft_client_verify_key
                output_blob = _serialize(output_policy, to_bytes=True)
                self.output_policy_state = output_blob
                return output_policy
            else:
                raise Exception("output_policy is None during init")

        try:
            output_policy = _deserialize(self.output_policy_state, from_bytes=True)
            output_policy.syft_server_location = self.syft_server_location
            output_policy.syft_client_verify_key = self.syft_client_verify_key
            return output_policy
        except Exception as e:
            print(f"Failed to deserialize custom output policy state. {e}")
            return None

    @property
    def output_policy_id(self) -> UID | None:
        if self.output_policy_init_kwargs is not None:
            return self.output_policy_init_kwargs.get("id", None)
        return None

    @property
    def input_policy_id(self) -> UID | None:
        if self.input_policy_init_kwargs is not None:
            return self.input_policy_init_kwargs.get("id", None)
        return None

    @output_policy.setter  # type: ignore
    def output_policy(self, value: Any) -> None:  # type: ignore
        if isinstance(value, OutputPolicy):
            self.output_policy_state = _serialize(value, to_bytes=True)
        elif (isinstance(value, bytes) and len(value) == 0) or value is None:
            self.output_policy_state = b""
        else:
            raise Exception(f"You can't set {type(value)} as output_policy_state")

    @property
    def output_history(self) -> list[ExecutionOutput]:
        api = self.get_api()
        return api.services.output.get_by_user_code_id(self.id)

    @as_result(SyftException)
    def get_output_history(
        self, context: AuthedServiceContext
    ) -> list[ExecutionOutput]:
        output_service = cast(
            OutputService, context.server.get_service("outputservice")
        )
        return output_service.get_by_user_code_id(context, self.id)

    @as_result(SyftException)
    def store_execution_output(
        self,
        context: AuthedServiceContext,
        outputs: Any,
        job_id: UID | None = None,
        input_ids: dict[str, UID] | None = None,
    ) -> ExecutionOutput:
        is_admin = context.role == ServiceRole.ADMIN

        output_policy = self.get_output_policy(context)

        if output_policy is None and not is_admin:
            raise SyftException(
                public_message="You must wait for the output policy to be approved"
            )

        output_ids = filter_only_uids(outputs)

        output_service = context.server.get_service("outputservice")
        output_service = cast(OutputService, output_service)
        return output_service.create(
            context,
            user_code_id=self.id,
            output_ids=output_ids,
            executing_user_verify_key=self.user_verify_key,
            job_id=job_id,
            output_policy_id=self.output_policy_id,
            input_ids=input_ids,
        )

    @property
    def byte_code(self) -> PyCodeObject | None:
        return compile_byte_code(self.parsed_code)

    @property
    def assets(self) -> DictTuple[str, Asset]:
        if not self.input_policy_init_kwargs:
            return DictTuple({})

        api = self._get_api()

        # get a flat dict of all inputs
        all_inputs = {}
        inputs = self.input_policy_init_kwargs or {}
        for vals in inputs.values():
            # Only keep UIDs, filter out Constants
            all_inputs.update({k: v for k, v in vals.items() if isinstance(v, UID)})

        # map the action_id to the asset
        used_assets: list[Asset] = []
        for kwarg_name, action_id in all_inputs.items():
            assets = api.dataset.get_assets_by_action_id(uid=action_id)
            if isinstance(assets, SyftError):
                return assets
            if assets:
                asset = assets[0]
                asset._kwarg_name = kwarg_name
                used_assets.append(asset)

        asset_dict = {asset._kwarg_name: asset for asset in used_assets}
        return DictTuple(asset_dict)

    @property
    def action_objects(self) -> dict:
        if not self.input_policy_init_kwargs:
            return {}

        all_inputs = {}
        for vals in self.input_policy_init_kwargs.values():
            all_inputs.update(vals)

        # filter out the assets
        action_objects = {
            arg_name: str(uid)
            for arg_name, uid in all_inputs.items()
            if arg_name not in self.assets.keys() and isinstance(uid, UID)
        }

        return action_objects

    @property
    def constants(self) -> dict[str, Constant]:
        if not self.input_policy_init_kwargs:
            return {}

        all_inputs = {}
        for vals in self.input_policy_init_kwargs.values():
            all_inputs.update(vals)

        # filter out the assets
        constants = {
            arg_name: item
            for arg_name, item in all_inputs.items()
            if isinstance(item, Constant)
        }

        return constants

    @property
    def inputs(self) -> dict:
        inputs = {}

        assets = self.assets
        action_objects = self.action_objects
        constants = self.constants
        if action_objects:
            inputs["action_objects"] = action_objects
        if assets:
            inputs["assets"] = {
                argument: asset._get_dict_for_user_code_repr()
                for argument, asset in assets.items()
            }
        if self.constants:
            inputs["constants"] = {
                argument: constant._get_dict_for_user_code_repr()
                for argument, constant in constants.items()
            }
        return inputs

    @property
    def _inputs_json(self) -> str | SyftError:
        input_str = json.dumps(self.inputs, indent=2)
        return input_str

    def get_sync_dependencies(self, context: AuthedServiceContext) -> list[UID]:
        dependencies = []

        if self.nested_codes is not None:
            nested_code_ids = [
                link.object_uid for link, _ in self.nested_codes.values()
            ]
            dependencies.extend(nested_code_ids)

        if self.status_link is not None:
            dependencies.append(self.status_link.object_uid)

        return dependencies

    @property
    def run(self) -> Callable | None:
        warning = SyftWarning(
            message="This code was submitted by a User and could be UNSAFE."
        )
        display(warning)

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
                    display(
                        SyftInfo(
                            message="The result you see is computed on PRIVATE data."
                        )
                    )
                if on_mock_data:
                    display(
                        SyftInfo(message="The result you see is computed on MOCK data.")
                    )

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
                raise SyftException(
                    public_message=f"Failed to execute 'run'. Error: {e}"
                )

        return wrapper

    @property
    @deprecated(reason="Use 'run' instead")
    def unsafe_function(self) -> Callable | None:
        return self.run

    def _inner_repr(self, level: int = 0) -> str:
        shared_with_line = ""
        if len(self.output_readers) > 0 and self.output_reader_names is not None:
            owners_string = " and ".join([f"*{x}*" for x in self.output_reader_names])
            shared_with_line += (
                f"Custom Policy: "
                f"outputs are *shared* with the owners of {owners_string} once computed"
            )

        constants_str = ""
        args = [
            x
            for _dict in self.input_policy_init_kwargs.values()  # type: ignore
            for x in _dict.values()
        ]
        constants = [x for x in args if isinstance(x, Constant)]
        constants_str = "\n\t".join([f"{x.kw}: {x.val}" for x in constants])

        # indent all lines except the first one
        inputs_str = "\n".join(
            [f"    {line}" for line in self._inputs_json.split("\n")]
        ).lstrip()

        md = f"""class UserCode
    id: UID = {self.id}
    service_func_name: str = {self.service_func_name}
    shareholders: list = {self.input_owners}
    status: list = {self.code_status}
    {constants_str}
    {shared_with_line}
    inputs: dict = {inputs_str}
    code:

{self.raw_code}
"""
        if self.nested_codes != {}:
            md += """

  Nested Requests:
  """

        md = "\n".join(
            [f"{'  '*level}{substring}" for substring in md.split("\n")[:-1]]
        )
        if self.nested_codes is not None:
            for obj, _ in self.nested_codes.values():
                code = obj.resolve
                md += "\n"
                md += code._inner_repr(level=level + 1)

        return md

    def _repr_markdown_(self, wrap_as_python: bool = True, indent: int = 0) -> str:
        return as_markdown_code(self._inner_repr())

    def _ipython_display_(self, level: int = 0) -> None:
        tabs = "&emsp;" * level
        shared_with_line = ""
        if len(self.output_readers) > 0 and self.output_reader_names is not None:
            owners_string = " and ".join([f"*{x}*" for x in self.output_reader_names])
            shared_with_line += (
                f"<p>{tabs}Custom Policy: "
                f"outputs are *shared* with the owners of {owners_string} once computed</p>"
            )
        constants_str = ""
        args = [
            x
            for _dict in self.input_policy_init_kwargs.values()  # type: ignore
            for x in _dict.values()
        ]
        constants = [x for x in args if isinstance(x, Constant)]
        constants_str = "\n&emsp;".join([f"{x.kw}: {x.val}" for x in constants])
        # indent all lines except the first one
        repr_str = f"""
    <style>
    {FONT_CSS}
    </style>
    <div class="syft-code">
    <h3>{tabs}UserCode</h3>
    <p>{tabs}<strong>id:</strong> UID = {self.id}</p>
    <p>{tabs}<strong>service_func_name:</strong> str = {self.service_func_name}</p>
    <p>{tabs}<strong>shareholders:</strong> list = {self.input_owners}</p>
    <p>{tabs}<strong>status:</strong> list = {self.code_status}</p>
    {tabs}{constants_str}
    {tabs}{shared_with_line}
    <p>{tabs}<strong>inputs:</strong> dict = <pre>{self._inputs_json}</pre></p>
    <p>{tabs}<strong>code:</strong></p>
    </div>
    """
        md = "\n".join(
            [f"{'  '*level}{substring}" for substring in self.raw_code.split("\n")[:-1]]
        )
        display(HTML(repr_str), Markdown(as_markdown_code(md)))
        if self.nested_codes is not None and self.nested_codes != {}:
            nested_line_html = f"""
    <div class="syft-code">
    <p>{tabs}<strong>Nested Requests:</p>
    </div>
    """
            display(HTML(nested_line_html))
            for obj, _ in self.nested_codes.values():
                code = obj.resolve
                code._ipython_display_(level=level + 1)

    @property
    def show_code(self) -> CodeMarkdown:
        return CodeMarkdown(self.raw_code)

    def show_code_cell(self) -> None:
        warning_message = """# WARNING: \n# Before you submit
# change the name of the function \n# for no duplicates\n\n"""

        # third party
        from IPython import get_ipython

        ip = get_ipython()
        ip.set_next_input(warning_message + self.raw_code)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        api = self._get_api()

        signature = self.signature
        signature = signature_remove_self(signature)
        signature = signature_remove_context(signature)
        remote_user_function = generate_remote_function(
            api=api,
            server_uid=self.server_uid,
            signature=self.signature,
            path="code.call",
            make_call=api.make_call,
            pre_kwargs={"uid": self.id},
            warning=None,
            communication_protocol=api.communication_protocol,
            unwrap_on_success=True,  # TODO: look into this
        )
        return remote_user_function(*args, **kwargs)


class UserCodeUpdate(PartialSyftObject):
    __canonical_name__ = "UserCodeUpdate"
    __version__ = SYFT_OBJECT_VERSION_1

    l0_deny_reason: str | None


@serializable(without=["local_function"])
class SubmitUserCode(SyftObject):
    # version
    __canonical_name__ = "SubmitUserCode"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID | None = None  # type: ignore[assignment]
    code: str
    func_name: str
    signature: inspect.Signature
    input_policy_type: SubmitUserPolicy | UID | type[InputPolicy]
    input_policy_init_kwargs: dict[Any, Any] | None = {}
    output_policy_type: SubmitUserPolicy | UID | type[OutputPolicy]
    output_policy_init_kwargs: dict[Any, Any] | None = {}
    local_function: Callable | None = None
    input_kwargs: list[str]
    worker_pool_name: str | None = None

    __repr_attrs__ = ["func_name", "code"]

    @field_validator("func_name", mode="after")
    @classmethod
    def func_name_is_valid(cls, value: str) -> str:
        _ = is_valid_usercode_name(
            value
        ).unwrap()  # this will throw an error if not valid
        return value

    @field_validator("output_policy_init_kwargs", mode="after")
    @classmethod
    def add_output_policy_ids(cls, values: Any) -> Any:
        if isinstance(values, dict) and "id" not in values:
            values["id"] = UID()
        return values

    @property
    def kwargs(self) -> dict[Any, Any] | None:
        return self.input_policy_init_kwargs

    def __call__(
        self,
        *args: Any,
        syft_no_server: bool = False,
        blocking: bool = False,
        time_alive: int | None = None,
        n_consumers: int = 2,
        **kwargs: Any,
    ) -> Any:
        if syft_no_server:
            return self.local_call(*args, **kwargs)
        return self._ephemeral_server_call(
            *args,
            time_alive=time_alive,
            n_consumers=n_consumers,
            blocking=blocking,
            **kwargs,
        )

    def local_call(self, *args: Any, **kwargs: Any) -> Any:
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

    def _ephemeral_server_call(
        self,
        *args: Any,
        time_alive: int | None = None,
        n_consumers: int = 2,
        blocking: bool = False,
        **kwargs: Any,
    ) -> Any:
        # relative
        from ...orchestra import Orchestra

        # Right now we only create a number of workers
        # In the future we might need to have the same pools/images as well

        if time_alive is None and not blocking:
            print(
                SyftInfo(
                    message="Closing the server after time_alive=300 (the default value)"
                )
            )
            time_alive = 300

        # This could be changed given the work on containers
        ep_server = Orchestra.launch(
            name=f"ephemeral_server_{self.func_name}_{random.randint(a=0, b=10000)}",  # nosec
            reset=True,
            create_producer=True,
            n_consumers=n_consumers,
            deploy_to="python",
        )
        ep_client = ep_server.login(
            email="info@openmined.org",
            password="changethis",
        )  # nosec
        self.input_policy_init_kwargs = cast(dict, self.input_policy_init_kwargs)
        for server_id, obj_dict in self.input_policy_init_kwargs.items():
            api = APIRegistry.get_by_recent_server_uid(server_uid=server_id.server_id)
            if api is None:
                raise SyftException(
                    public_message=f"Can't access the api. You must login to {server_id.server_id}"
                )
            # Creating TwinObject from the ids of the kwargs
            # Maybe there are some corner cases where this is not enough
            # And need only ActionObjects
            # Also, this works only on the assumption that all inputs
            # are ActionObjects, which might change in the future
            for id in obj_dict.values():
                try:
                    mock_obj = api.services.action.get_mock(id)
                    data_obj = mock_obj
                except SyftException:
                    try:
                        data_obj = api.services.action.get(id)
                    except SyftException:
                        raise SyftException(
                            public_message="You do not have access to object you want \
                                to use, or the private object does not have mock \
                                data. Contact the Server Admin."
                        )

                data_obj.id = id
                new_obj = ActionObject.from_obj(
                    data_obj.syft_action_data,
                    id=id,
                    syft_server_location=server_id.server_id,
                    syft_client_verify_key=server_id.verify_key,
                )
                new_obj.send(ep_client)

        new_syft_func = deepcopy(self)

        # This will only be used without worker_pools
        new_syft_func.worker_pool_name = None

        # We will look for subjos, and if we find any will submit them
        # to the ephemeral_server
        submit_subjobs_code(self, ep_client)

        ep_client.code.request_code_execution(new_syft_func)
        ep_client.requests[-1].approve(approve_nested=True)
        func_call = getattr(ep_client.code, new_syft_func.func_name)
        # TODO: fix properly
        func_call.unwrap_on_success = True
        result = func_call(*args, **kwargs)

        def task() -> None:
            if "blocking" in kwargs and not kwargs["blocking"]:
                time.sleep(time_alive)
            print(SyftInfo(message="Landing the ephmeral server..."))
            ep_server.land()
            print(SyftInfo(message="Server Landed!"))

        thread = Thread(target=task)
        thread.start()

        return result

    @property
    def input_owner_verify_keys(self) -> list[str] | None:
        if self.input_policy_init_kwargs is not None:
            return [x.verify_key for x in self.input_policy_init_kwargs.keys()]
        return None


def get_code_hash(code: str, user_verify_key: SyftVerifyKey) -> str:
    full_str = f"{code}{user_verify_key}"
    return hashlib.sha256(full_str.encode()).hexdigest()


@as_result(SyftException)
def is_valid_usercode_name(func_name: str) -> Any:
    if len(func_name) == 0:
        raise SyftException(public_message="Function name cannot be empty")
    if func_name == "_":
        raise SyftException(
            public_message="Cannot use anonymous function as syft function"
        )
    if not str.isidentifier(func_name):
        raise SyftException(
            public_message="Function name must be a valid Python identifier"
        )
    if keyword.iskeyword(func_name):
        raise SyftException(public_message="Function name is a reserved python keyword")

    service_method_path = f"code.{func_name}"
    if ServiceConfigRegistry.path_exists(service_method_path):
        raise SyftException(
            public_message=(
                f"Could not create syft function with name {func_name}:"
                f" a service with the same name already exists"
            )
        )
    return True


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


def syft_function_single_use(
    *args: Any,
    share_results_with_owners: bool = False,
    worker_pool_name: str | None = None,
    **kwargs: Any,
) -> Callable:
    return syft_function(
        input_policy=ExactMatch(*args, **kwargs),
        output_policy=SingleExecutionExactOutput(),
        share_results_with_owners=share_results_with_owners,
        worker_pool_name=worker_pool_name,
    )


def replace_func_name(src: str, new_func_name: str) -> str:
    pattern = r"\bdef\s+(\w+)\s*\("
    replacement = f"def {new_func_name}("
    new_src = re.sub(pattern, replacement, src, count=1)
    return new_src


def syft_function(
    input_policy: InputPolicy | UID | None = None,
    output_policy: OutputPolicy | UID | None = None,
    share_results_with_owners: bool = False,
    worker_pool_name: str | None = None,
    name: str | None = None,
) -> Callable:
    if input_policy is None:
        input_policy = EmpyInputPolicy()

    init_input_kwargs = None
    if isinstance(input_policy, CustomInputPolicy):
        input_policy_type = SubmitUserPolicy.from_obj(input_policy)
        init_input_kwargs = partition_by_server(input_policy.init_kwargs)  # type: ignore
    else:
        input_policy_type = type(input_policy)
        init_input_kwargs = getattr(input_policy, "init_kwargs", {})

    if output_policy is None:
        output_policy = SingleExecutionExactOutput()

    if isinstance(output_policy, CustomOutputPolicy):
        output_policy_type = SubmitUserPolicy.from_obj(output_policy)
    else:
        output_policy_type = type(output_policy)

    def decorator(f: Any) -> SubmitUserCode:
        try:
            code = dedent(inspect.getsource(f))

            if name is not None:
                fname = name
                code = replace_func_name(code, fname)
            else:
                fname = f.__name__

            input_kwargs = f.__code__.co_varnames[: f.__code__.co_argcount]

            parse_user_code(
                raw_code=code,
                func_name=fname,
                original_func_name=f.__name__,
                function_input_kwargs=input_kwargs,
            )

            res = SubmitUserCode(
                code=code,
                func_name=fname,
                signature=inspect.signature(f),
                input_policy_type=input_policy_type,
                input_policy_init_kwargs=init_input_kwargs,
                output_policy_type=output_policy_type,
                output_policy_init_kwargs=getattr(output_policy, "init_kwargs", {}),
                local_function=f,
                input_kwargs=input_kwargs,
                worker_pool_name=worker_pool_name,
            )

        except ValidationError as e:
            errors = e.errors()
            msg = "Failed to create syft function, encountered validation errors:\n"
            for error in errors:
                msg += f"\t{error['msg']}\n"
            raise SyftException(public_message=msg)

        except SyftException as se:
            raise SyftException(public_message=f"Error when parsing the code: {se}")

        if share_results_with_owners and res.output_policy_init_kwargs is not None:
            res.output_policy_init_kwargs["output_readers"] = (
                res.input_owner_verify_keys
            )

        success_message = SyftSuccess(
            message=f"Syft function '{f.__name__}' successfully created. "
            f"To add a code request, please create a project using `project = syft.Project(...)`, "
            f"then use command `project.create_code_request`."
        )
        display(success_message)

        return res

    return decorator


def generate_unique_func_name(context: TransformContext) -> TransformContext:
    if context.output is not None:
        code_hash = context.output["code_hash"]
        service_func_name = context.output["func_name"]
        context.output["service_func_name"] = service_func_name
        func_name = f"user_func_{service_func_name}_{context.credentials}_{code_hash}"
        user_unique_func_name = (
            f"user_func_{service_func_name}_{context.credentials}_{time.time()}"
        )
        context.output["unique_func_name"] = func_name
        context.output["user_unique_func_name"] = user_unique_func_name
    return context


def parse_user_code(
    raw_code: str,
    func_name: str,
    original_func_name: str,
    function_input_kwargs: list[str],
) -> str:
    # parse the code, check for syntax errors and if there are global variables
    tree: ast.Module = parse_code(raw_code=raw_code)
    check_for_global_vars(code_tree=tree)

    f: ast.stmt = tree.body[0]
    f.decorator_list = []

    call_args = function_input_kwargs
    call_stmt_keywords = [ast.keyword(arg=i, value=[ast.Name(id=i)]) for i in call_args]
    call_stmt = ast.Assign(
        targets=[ast.Name(id="result")],
        value=ast.Call(
            func=ast.Name(id=original_func_name), args=[], keywords=call_stmt_keywords
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


def process_code(
    context: TransformContext,
    raw_code: str,
    func_name: str,
    original_func_name: str,
    policy_input_kwargs: list[str],
    function_input_kwargs: list[str],
) -> str:
    if "datasite" in function_input_kwargs and context.output is not None:
        context.output["uses_datasite"] = True

    return parse_user_code(
        raw_code=raw_code,
        func_name=func_name,
        original_func_name=original_func_name,
        function_input_kwargs=function_input_kwargs,
    )


def new_check_code(context: TransformContext) -> TransformContext:
    # TODO: remove this tech debt hack
    if context.output is None:
        return context

    input_kwargs = context.output["input_policy_init_kwargs"]
    server_view_workaround = False
    for k in input_kwargs.keys():
        if isinstance(k, ServerIdentity):
            server_view_workaround = True

    if not server_view_workaround:
        input_keys = list(input_kwargs.keys())
    else:
        input_keys = []
        for d in input_kwargs.values():
            input_keys += d.keys()

    processed_code = process_code(
        context,
        raw_code=context.output["raw_code"],
        func_name=context.output["unique_func_name"],
        original_func_name=context.output["service_func_name"],
        policy_input_kwargs=input_keys,
        function_input_kwargs=context.output["input_kwargs"],
    )
    context.output["parsed_code"] = processed_code

    return context


def locate_launch_jobs(context: TransformContext) -> TransformContext:
    if context.server is None:
        raise ValueError(f"context {context}'s server is None")
    if context.output is not None:
        nested_codes = {}
        tree = ast.parse(context.output["raw_code"])
        # look for datasite arg
        if "datasite" in [arg.arg for arg in tree.body[0].args.args]:
            v = LaunchJobVisitor()
            v.visit(tree)
            nested_calls = v.nested_calls
            user_code_service = context.server.get_service("usercodeService")
            for call in nested_calls:
                user_codes = user_code_service.get_by_service_name(context, call)
                # TODO: Not great
                user_code = user_codes[-1]
                user_code_link = LinkedObject.from_obj(
                    user_code, server_uid=context.server.id
                )
                nested_codes[call] = (user_code_link, user_code.nested_codes)
        context.output["nested_codes"] = nested_codes
    return context


def compile_code(context: TransformContext) -> TransformContext:
    if context.output is None:
        return context

    byte_code = compile_byte_code(context.output["parsed_code"])
    if byte_code is None:
        raise ValueError(
            "Unable to compile byte code from parsed code. "
            + context.output["parsed_code"]
        )
    return context


def hash_code(context: TransformContext) -> TransformContext:
    if context.output is None:
        return context
    if not isinstance(context.obj, SubmitUserCode):
        return context

    code = context.output["code"]
    context.output["raw_code"] = code
    code_hash = get_code_hash(code, context.credentials)
    context.output["code_hash"] = code_hash

    return context


def add_credentials_for_key(key: str) -> Callable:
    def add_credentials(context: TransformContext) -> TransformContext:
        if context.output is not None:
            context.output[key] = context.credentials
        return context

    return add_credentials


def check_policy(policy: Any, context: TransformContext) -> TransformContext:
    if context.server is not None:
        policy_service = context.server.get_service(PolicyService)
        if isinstance(policy, SubmitUserPolicy):
            policy = policy.to(UserPolicy, context=context)
        elif isinstance(policy, UID):
            policy = policy_service.get_policy_by_uid(context, policy)
    return policy


def check_input_policy(context: TransformContext) -> TransformContext:
    if context.output is None:
        return context

    ip = context.output["input_policy_type"]
    ip = check_policy(policy=ip, context=context)
    context.output["input_policy_type"] = ip

    return context


def check_output_policy(context: TransformContext) -> TransformContext:
    if context.output is not None:
        op = context.output["output_policy_type"]
        op = check_policy(policy=op, context=context)
        context.output["output_policy_type"] = op
    return context


def create_code_status(context: TransformContext) -> TransformContext:
    # relative
    from .user_code_service import UserCodeService

    if context.server is None:
        raise ValueError(f"{context}'s server is None")

    if context.output is None:
        return context

    # Low side requests have a computed status
    if context.server.server_side_type == ServerSideType.LOW_SIDE:
        return context

    input_keys = list(context.output["input_policy_init_kwargs"].keys())
    code_link = LinkedObject.from_uid(
        context.output["id"],
        UserCode,
        service_type=UserCodeService,
        server_uid=context.server.id,
    )
    if context.server.server_type == ServerType.DATASITE:
        server_identity = ServerIdentity(
            server_name=context.server.name,
            server_id=context.server.id,
            verify_key=context.server.signing_key.verify_key,
        )
        status = UserCodeStatusCollection(
            status_dict={server_identity: (UserCodeStatus.PENDING, "")},
            user_code_link=code_link,
        )

    elif context.server.server_type == ServerType.ENCLAVE:
        status_dict = {key: (UserCodeStatus.PENDING, "") for key in input_keys}
        status = UserCodeStatusCollection(
            status_dict=status_dict,
            user_code_link=code_link,
        )
    else:
        raise NotImplementedError(
            f"Invalid server type:{context.server.server_type} for code submission"
        )

    res = context.server.get_service("usercodestatusservice").create(context, status)
    # relative
    from .status_service import UserCodeStatusService

    context.output["status_link"] = LinkedObject.from_uid(
        res.id,
        UserCodeStatusCollection,
        service_type=UserCodeStatusService,
        server_uid=context.server.id,
    )
    return context


def add_submit_time(context: TransformContext) -> TransformContext:
    if context.output:
        context.output["submit_time"] = DateTime.now()
    return context


def set_default_pool_if_empty(context: TransformContext) -> TransformContext:
    if (
        context.server
        and context.output
        and context.output.get("worker_pool_name", None) is None
    ):
        default_pool = context.server.get_default_worker_pool().unwrap()
        context.output["worker_pool_name"] = default_pool.name
    return context


def set_origin_server_side_type(context: TransformContext) -> TransformContext:
    if context.server and context.output:
        context.output["origin_server_side_type"] = (
            context.server.server_side_type or ServerSideType.HIGH_SIDE
        )
    return context


@transform(SubmitUserCode, UserCode)
def submit_user_code_to_user_code() -> list[Callable]:
    return [
        generate_id,
        hash_code,
        generate_unique_func_name,
        check_input_policy,
        check_output_policy,
        new_check_code,
        locate_launch_jobs,
        add_credentials_for_key("user_verify_key"),
        create_code_status,
        add_server_uid_for_key("server_uid"),
        add_submit_time,
        set_default_pool_if_empty,
        set_origin_server_side_type,
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
    result: Any = None


@serializable()
class UserCodeExecutionOutputV1(SyftObject):
    # version
    __canonical_name__ = "UserCodeExecutionOutput"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    user_code_id: UID
    stdout: str
    stderr: str
    result: Any = None


@serializable()
class UserCodeExecutionOutput(SyftObject):
    # version
    __canonical_name__ = "UserCodeExecutionOutput"
    __version__ = SYFT_OBJECT_VERSION_2

    id: UID
    user_code_id: UID
    errored: bool = False
    stdout: str
    stderr: str
    result: Any = None
    safe_error_message: str | None = None


class SecureContext:
    def __init__(self, context: AuthedServiceContext) -> None:
        server = context.server
        if server is None:
            raise ValueError(f"{context}'s server is None")

        job_service = server.get_service("jobservice")
        action_service = server.get_service("actionservice")
        # user_service = server.get_service("userservice")

        def job_set_n_iters(n_iters: int) -> None:
            job = context.job
            job.n_iters = n_iters
            job_service.update(context, job)

        def job_set_current_iter(current_iter: int) -> None:
            job = context.job
            job.current_iter = current_iter
            job_service.update(context, job)

        def job_increase_current_iter(current_iter: int) -> None:
            job = context.job
            job.current_iter += current_iter
            job_service.update(context, job)

        # def set_api_registry():
        #     user_signing_key = [
        #         x.signing_key
        #         for x in user_service.stash.partition.data.values()
        #         if x.verify_key == context.credentials
        #     ][0]
        #     data_protcol = get_data_protocol()
        #     user_api = server.get_api(context.credentials, data_protcol.latest_version)
        #     user_api.signing_key = user_signing_key
        #     # We hardcode a python connection here since we have access to the server
        #     # TODO: this is not secure
        #     user_api.connection = PythonConnection(server=server)

        #     APIRegistry.set_api_for(
        #         server_uid=server.id,
        #         user_verify_key=context.credentials,
        #         api=user_api,
        #     )

        def launch_job(func: UserCode, **kwargs: Any) -> Job | None:
            # relative

            kw2id = {}
            for k, v in kwargs.items():
                value = ActionObject.from_obj(v)
                ptr = action_service.set_result_to_store(
                    value, context, has_result_read_permission=False
                ).unwrap()
                kw2id[k] = ptr.id
            try:
                # TODO: check permissions here
                action = Action.syft_function_action_from_kwargs_and_id(kw2id, func.id)

                return server.add_action_to_queue(
                    action=action,
                    credentials=context.credentials,
                    parent_job_id=context.job_id,
                    has_execute_permissions=True,
                    worker_pool_name=func.worker_pool_name,
                ).unwrap()
                # # set api in global scope to enable using .get(), .wait())
                # set_api_registry()
            except Exception as e:
                print(f"ERROR {e}")
                raise ValueError(f"error while launching job:\n{e}")

        self.job_set_n_iters = job_set_n_iters
        self.job_set_current_iter = job_set_current_iter
        self.job_increase_current_iter = job_increase_current_iter
        self.launch_job = launch_job
        self.is_async = context.job is not None


def execute_byte_code(
    code_item: UserCode, kwargs: dict[str, Any], context: AuthedServiceContext
) -> Any:
    stdout_ = sys.stdout
    stderr_ = sys.stderr

    try:
        # stdlib
        import builtins as __builtin__

        original_print = __builtin__.print

        safe_context = SecureContext(context=context)

        class LocalDatasiteClient:
            def init_progress(self, n_iters: int) -> None:
                if safe_context.is_async:
                    safe_context.job_set_current_iter(0)
                    safe_context.job_set_n_iters(n_iters)

            def set_progress(self, to: int) -> None:
                self._set_progress(to)

            def increment_progress(self, n: int = 1) -> None:
                self._set_progress(by=n)

            def _set_progress(
                self, to: int | None = None, by: int | None = None
            ) -> None:
                if safe_context.is_async is not None:
                    if by is None and to is None:
                        by = 1
                    if to is None:
                        safe_context.job_increase_current_iter(current_iter=by)
                    else:
                        safe_context.job_set_current_iter(to)

            @final
            def launch_job(self, func: UserCode, **kwargs: Any) -> Job | None:
                return safe_context.launch_job(func, **kwargs)

            def __setattr__(self, __name: str, __value: Any) -> None:
                raise Exception("Attempting to alter read-only value")

        if context.job is not None:
            job_id = context.job_id
            log_id = context.job.log_id

            def print(*args: Any, sep: str = " ", end: str = "\n") -> str | None:
                def to_str(arg: Any) -> str:
                    if isinstance(arg, bytes):
                        return arg.decode("utf-8")
                    if isinstance(arg, Job):
                        return f"JOB: {arg.id}"
                    if isinstance(arg, SyftError):
                        return f"JOB: {arg.message}"
                    if isinstance(arg, ActionObject):
                        return str(arg.syft_action_data)
                    return str(arg)

                new_args = [to_str(arg) for arg in args]
                new_str = sep.join(new_args) + end
                if context.server is not None:
                    log_service = context.server.get_service("LogService")
                    log_service.append(context=context, uid=log_id, new_str=new_str)
                time = datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")
                return __builtin__.print(
                    f"{time} FUNCTION LOG ({job_id}):",
                    *new_args,
                    end=end,
                    sep=sep,
                    file=sys.stderr,
                )

        else:
            print = original_print

        if code_item.uses_datasite:
            kwargs["datasite"] = LocalDatasiteClient()

        job_log_id = context.job.log_id if context.job else None
        for k, v in kwargs.items():
            if isinstance(v, CustomEndpointActionObject):
                kwargs[k] = v.add_context(context=context, log_id=job_log_id)

        stdout = StringIO()
        stderr = StringIO()

        # statisfy lint checker
        result = None

        # We only need access to local kwargs
        _locals = {"kwargs": kwargs}
        _globals = {}

        if code_item.nested_codes is not None:
            for service_func_name, (linked_obj, _) in code_item.nested_codes.items():
                _globals[service_func_name] = linked_obj.resolve_with_context(
                    context=context
                ).unwrap()

        _globals["print"] = print
        exec(code_item.parsed_code, _globals, _locals)  # nosec

        evil_string = f"{code_item.unique_func_name}(**kwargs)"

        result_message = ""

        try:
            result = eval(evil_string, _globals, _locals)  # nosec
            errored = False
        except Exception as e:
            errored = True
            error_msg = traceback_from_error(e, code_item)

            if context.job is not None:
                time = datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")
                logger.error(f"{time} EXCEPTION LOG ({job_id}):\n{error_msg}")
            else:
                # for local execution
                time = datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")
                logger.error(f"{time} EXCEPTION LOG:\n{error_msg}\n")

            if (
                context.server is not None
                and context.job is not None
                and context.job.log_id is not None
            ):
                log_id = context.job.log_id
                log_service = context.server.get_service("LogService")
                log_service.append(context=context, uid=log_id, new_err=error_msg)

            result_message = (
                f"Exception encountered while running {code_item.service_func_name}"
                ", please contact the Server Admin for more info."
            )

            if context.dev_mode:
                result_message += error_msg

            result = SyftError(message=result_message)

        # reset print
        print = original_print

        # restore stdout and stderr
        sys.stdout = stdout_
        sys.stderr = stderr_

        return UserCodeExecutionOutput(
            user_code_id=code_item.id,
            stdout=str(stdout.getvalue()),
            stderr=str(stderr.getvalue()),
            result=result,
            errored=errored,
            safe_error_message=result_message,
        )
    except Exception as e:
        # stdlib

        print = original_print
        # print("execute_byte_code failed", e, file=stderr_)
        print(traceback.format_exc())
        print("execute_byte_code failed", e)
        raise
    finally:
        sys.stdout = stdout_
        sys.stderr = stderr_


def traceback_from_error(e: Exception, code: UserCode) -> str:
    """We do this because the normal traceback.format_exc() does not work well for exec,
    it missed the references to the actual code"""
    line_nr = 0
    tb = e.__traceback__
    while tb is not None:
        line_nr = tb.tb_lineno - 1
        tb = tb.tb_next

    lines = code.parsed_code.split("\n")
    start_line = max(0, line_nr - 2)
    end_line = min(len(lines), line_nr + 2)
    error_lines: list[str] | str = [
        (
            e.replace("   ", f"    {i} ", 1)
            if i != line_nr
            else e.replace("   ", f"--> {i} ", 1)
        )
        for i, e in enumerate(lines)
        if i >= start_line and i < end_line
    ]
    error_lines = "\n".join(error_lines)

    error_msg = f"""
Encountered while executing {code.service_func_name}:
{traceback.format_exc()}
{error_lines}"""
    return error_msg


def load_approved_policy_code(
    user_code_items: list[UserCode], context: AuthedServiceContext | None
) -> Any:
    """Reload the policy code in memory for user code that is approved."""
    for user_code in user_code_items:
        try:
            if context is None:
                status = user_code.status
            else:
                status = user_code.get_status(context).unwrap()
        except SyftException:
            display(
                SyftWarning(
                    message=f"Failed to load UserCode {user_code.id.no_dash} {user_code.service_func_name=}"
                )
            )
            continue

        if status.approved:
            if isinstance(user_code.input_policy_type, UserPolicy):
                load_policy_code(user_code.input_policy_type)
            if isinstance(user_code.output_policy_type, UserPolicy):
                load_policy_code(user_code.output_policy_type)
