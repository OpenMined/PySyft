# future
from __future__ import annotations

# stdlib
import ast
from collections.abc import Callable
from collections.abc import Generator
from copy import deepcopy
import datetime
from enum import Enum
import hashlib
import inspect
from io import StringIO
import itertools
import random
import sys
from threading import Thread
import time
import traceback
from typing import Any
from typing import ClassVar
from typing import TYPE_CHECKING
from typing import cast
from typing import final

# third party
from IPython.display import display
from pydantic import field_validator
from result import Err
from typing_extensions import Self

# relative
from ...abstract_node import NodeType
from ...client.api import APIRegistry
from ...client.api import NodeIdentity
from ...client.enclave_client import EnclaveMetadata
from ...node.credentials import SyftVerifyKey
from ...serde.deserialize import _deserialize
from ...serde.serializable import serializable
from ...serde.serialize import _serialize
from ...store.document_store import PartitionKey
from ...store.linked_obj import LinkedObject
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SYFT_OBJECT_VERSION_4
from ...types.syft_object import SyftObject
from ...types.syncable_object import SyncableSyftObject
from ...types.transforms import TransformContext
from ...types.transforms import add_node_uid_for_key
from ...types.transforms import generate_id
from ...types.transforms import transform
from ...types.uid import UID
from ...util import options
from ...util.colors import SURFACE
from ...util.markdown import CodeMarkdown
from ...util.markdown import as_markdown_code
from ..action.action_endpoint import CustomEndpointActionObject
from ..action.action_object import Action
from ..action.action_object import ActionObject
from ..context import AuthedServiceContext
from ..dataset.dataset import Asset
from ..job.job_stash import Job
from ..output.output_service import ExecutionOutput
from ..output.output_service import OutputService
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
from ..policy.policy_service import PolicyService
from ..response import SyftError
from ..response import SyftInfo
from ..response import SyftNotReady
from ..response import SyftSuccess
from ..response import SyftWarning
from .code_parse import GlobalsVisitor
from .code_parse import LaunchJobVisitor
from .unparse import unparse
from .utils import submit_subjobs_code

if TYPE_CHECKING:
    # relative
    from ...service.sync.diff_state import AttrDiff

UserVerifyKeyPartitionKey = PartitionKey(key="user_verify_key", type_=SyftVerifyKey)
CodeHashPartitionKey = PartitionKey(key="code_hash", type_=str)
ServiceFuncNamePartitionKey = PartitionKey(key="service_func_name", type_=str)
SubmitTimePartitionKey = PartitionKey(key="submit_time", type_=DateTime)

PyCodeObject = Any


@serializable()
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

    status_dict: dict[NodeIdentity, tuple[UserCodeStatus, str]] = {}
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
        string = f"""
            <style>
                .syft-user_code {{color: {SURFACE[options.color_theme]};}}
                </style>
                <div class='syft-user_code'>
                    <h3 style="line-height: 25%; margin-top: 25px;">User Code Status</h3>
                    <p style="margin-left: 3px;">
            """
        for node_identity, (status, reason) in self.status_dict.items():
            node_name_str = f"{node_identity.node_name}"
            uid_str = f"{node_identity.node_id}"
            status_str = f"{status.value}"
            string += f"""
                    &#x2022; <strong>UID: </strong>{uid_str}&nbsp;
                    <strong>Node name: </strong>{node_name_str}&nbsp;
                    <strong>Status: </strong>{status_str};
                    <strong>Reason: </strong>{reason}
                    <br>
                """
        string += "</p></div>"
        return string

    def __repr_syft_nested__(self) -> str:
        string = ""
        for node_identity, (status, reason) in self.status_dict.items():
            string += f"{node_identity.node_name}: {status}, {reason}<br>"
        return string

    def get_status_message(self) -> SyftSuccess | SyftNotReady | SyftError:
        if self.approved:
            return SyftSuccess(message=f"{type(self)} approved")
        denial_string = ""
        string = ""
        for node_identity, (status, reason) in self.status_dict.items():
            denial_string += f"Code status on node '{node_identity.node_name}' is '{status}'. Reason: {reason}"
            if not reason.endswith("."):
                denial_string += "."
            string += f"Code status on node '{node_identity.node_name}' is '{status}'."
        if self.denied:
            return SyftError(
                message=f"{type(self)} Your code cannot be run: {denial_string}"
            )
        else:
            return SyftNotReady(
                message=f"{type(self)} Your code is waiting for approval. {string}"
            )

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
        if context.node.node_type == NodeType.ENCLAVE:
            keys = {status for status, _ in self.status_dict.values()}
            if len(keys) == 1 and UserCodeStatus.APPROVED in keys:
                return UserCodeStatus.APPROVED
            elif UserCodeStatus.PENDING in keys and UserCodeStatus.DENIED not in keys:
                return UserCodeStatus.PENDING
            elif UserCodeStatus.DENIED in keys:
                return UserCodeStatus.DENIED
            else:
                raise Exception(f"Invalid types in {keys} for Code Submission")

        elif context.node.node_type == NodeType.DOMAIN:
            node_identity = NodeIdentity(
                node_name=context.node.name,
                node_id=context.node.id,
                verify_key=context.node.signing_key.verify_key,
            )
            if node_identity in self.status_dict:
                return self.status_dict[node_identity][0]
            else:
                raise Exception(
                    f"Code Object does not contain {context.node.name} Domain's data"
                )
        else:
            raise Exception(
                f"Invalid Node Type for Code Submission:{context.node.node_type}"
            )

    def mutate(
        self,
        value: tuple[UserCodeStatus, str],
        node_name: str,
        node_id: UID,
        verify_key: SyftVerifyKey,
    ) -> SyftError | Self:
        node_identity = NodeIdentity(
            node_name=node_name, node_id=node_id, verify_key=verify_key
        )
        status_dict = self.status_dict
        if node_identity in status_dict:
            status_dict[node_identity] = value
            self.status_dict = status_dict
            return self
        else:
            return SyftError(
                message="Cannot Modify Status as the Domain's data is not included in the request"
            )

    def get_sync_dependencies(self, context: AuthedServiceContext) -> list[UID]:
        return [self.user_code_link.object_uid]


@serializable()
class UserCode(SyncableSyftObject):
    # version
    __canonical_name__ = "UserCode"
    __version__ = SYFT_OBJECT_VERSION_4

    id: UID
    node_uid: UID | None = None
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
    status_link: LinkedObject
    input_kwargs: list[str]
    enclave_metadata: EnclaveMetadata | None = None
    submit_time: DateTime | None = None
    uses_domain: bool = False  # tracks if the code calls domain.something, variable is set during parsing
    nested_codes: dict[str, tuple[LinkedObject, dict]] | None = {}
    worker_pool_name: str | None = None

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
    ]

    __exclude_sync_diff_attrs__: ClassVar[list[str]] = [
        "node_uid",
        "input_policy_type",
        "input_policy_init_kwargs",
        "input_policy_state",
        "output_policy_type",
        "output_policy_init_kwargs",
        "output_policy_state",
    ]

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
    def status(self) -> UserCodeStatusCollection | SyftError:
        # Clientside only
        res = self.status_link.resolve
        return res

    def get_status(
        self, context: AuthedServiceContext
    ) -> UserCodeStatusCollection | SyftError:
        status = self.status_link.resolve_with_context(context)
        if status.is_err():
            return SyftError(message=status.err())
        return status.ok()

    @property
    def is_enclave_code(self) -> bool:
        return self.enclave_metadata is not None

    @property
    def input_owners(self) -> list[str] | None:
        if self.input_policy_init_kwargs is not None:
            return [str(x.node_name) for x in self.input_policy_init_kwargs.keys()]
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
                x.verify_key: x.node_name for x in self.input_policy_init_kwargs
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
        for node_view, (status, _) in self.status.status_dict.items():
            status_list.append(
                f"Node: {node_view.node_name}, Status: {status.value}",
            )
        return status_list

    @property
    def input_policy(self) -> InputPolicy | None:
        if not self.status.approved:
            return None
        return self._get_input_policy()

    def get_input_policy(self, context: AuthedServiceContext) -> InputPolicy | None:
        status = self.get_status(context)
        if not status.approved:
            return None
        return self._get_input_policy()

    def _get_input_policy(self) -> InputPolicy | None:
        if len(self.input_policy_state) == 0:
            input_policy = None
            if (
                isinstance(self.input_policy_type, type)
                and issubclass(self.input_policy_type, InputPolicy)
                and self.input_policy_init_kwargs is not None
            ):
                # TODO: Tech Debt here
                node_view_workaround = False
                for k, _ in self.input_policy_init_kwargs.items():
                    if isinstance(k, NodeIdentity):
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

    def is_output_policy_approved(self, context: AuthedServiceContext) -> bool:
        return self.get_status(context).approved

    @input_policy.setter  # type: ignore
    def input_policy(self, value: Any) -> None:  # type: ignore
        if isinstance(value, InputPolicy):
            self.input_policy_state = _serialize(value, to_bytes=True)
        elif (isinstance(value, bytes) and len(value) == 0) or value is None:
            self.input_policy_state = b""
        else:
            raise Exception(f"You can't set {type(value)} as input_policy_state")

    def get_output_policy(self, context: AuthedServiceContext) -> OutputPolicy | None:
        if not self.get_status(context).approved:
            return None
        return self._get_output_policy()

    def _get_output_policy(self) -> OutputPolicy | None:
        # if not self.status.approved:
        #     return None
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
                output_policy.syft_node_location = self.syft_node_location
                output_policy.syft_client_verify_key = self.syft_client_verify_key
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

    @property
    def output_policy(self) -> OutputPolicy | None:  # type: ignore
        if not self.status.approved:
            return None
        return self._get_output_policy()

    @output_policy.setter  # type: ignore
    def output_policy(self, value: Any) -> None:  # type: ignore
        if isinstance(value, OutputPolicy):
            self.output_policy_state = _serialize(value, to_bytes=True)
        elif (isinstance(value, bytes) and len(value) == 0) or value is None:
            self.output_policy_state = b""
        else:
            raise Exception(f"You can't set {type(value)} as output_policy_state")

    @property
    def output_history(self) -> list[ExecutionOutput] | SyftError:
        api = APIRegistry.api_for(self.syft_node_location, self.syft_client_verify_key)
        if api is None:
            return SyftError(
                message=f"Can't access the api. You must login to {self.syft_node_location}"
            )
        return api.services.output.get_by_user_code_id(self.id)

    def get_output_history(
        self, context: AuthedServiceContext
    ) -> list[ExecutionOutput] | SyftError:
        if not self.get_status(context).approved:
            return SyftError(
                message="Execution denied, Please wait for the code to be approved"
            )

        output_service = cast(OutputService, context.node.get_service("outputservice"))
        return output_service.get_by_user_code_id(context, self.id)

    def store_as_history(
        self,
        context: AuthedServiceContext,
        outputs: Any,
        job_id: UID | None = None,
        input_ids: dict[str, UID] | None = None,
    ) -> ExecutionOutput | SyftError:
        output_policy = self.get_output_policy(context)
        if output_policy is None:
            return SyftError(
                message="You must wait for the output policy to be approved"
            )

        output_ids = filter_only_uids(outputs)

        output_service = context.node.get_service("outputservice")
        output_service = cast(OutputService, output_service)
        execution_result = output_service.create(
            context,
            user_code_id=self.id,
            output_ids=output_ids,
            executing_user_verify_key=self.user_verify_key,
            job_id=job_id,
            output_policy_id=output_policy.id,
            input_ids=input_ids,
        )
        if isinstance(execution_result, SyftError):
            return execution_result

        return execution_result

    @property
    def byte_code(self) -> PyCodeObject | None:
        return compile_byte_code(self.parsed_code)

    def get_results(self) -> Any:
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(self.node_uid, self.syft_client_verify_key)
        if api is None:
            return SyftError(
                message=f"Can't access the api. You must login to {self.node_uid}"
            )
        return api.services.code.get_results(self)

    @property
    def assets(self) -> list[Asset]:
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(self.node_uid, self.syft_client_verify_key)
        if api is None:
            return SyftError(message=f"You must login to {self.node_uid}")

        inputs: Generator = (x for x in range(0))  # create an empty generator
        if self.input_policy_init_kwargs is not None:
            inputs = (
                uids
                for node_identity, uids in self.input_policy_init_kwargs.items()
                if node_identity.node_name == api.node_name
            )

        all_assets = []
        for uid in itertools.chain.from_iterable(x.values() for x in inputs):
            if isinstance(uid, UID):
                assets = api.services.dataset.get_assets_by_action_id(uid)
                if not isinstance(assets, list):
                    return assets

                all_assets += assets
        return all_assets

    def get_sync_dependencies(
        self, context: AuthedServiceContext
    ) -> list[UID] | SyftError:
        dependencies = []

        if self.nested_codes is not None:
            nested_code_ids = [
                link.object_uid for link, _ in self.nested_codes.values()
            ]
            dependencies.extend(nested_code_ids)

        dependencies.append(self.status_link.object_uid)

        return dependencies

    @property
    def unsafe_function(self) -> Callable | None:
        warning = SyftWarning(
            message="This code was submitted by a User and could be UNSAFE."
        )
        display(warning)

        # 🟡 TODO: re-use the same infrastructure as the execute_byte_code function
        def wrapper(*args: Any, **kwargs: Any) -> Callable | SyftError:
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
                return SyftError(f"Failed to run unsafe_function. Error: {e}")

        return wrapper

    def _inner_repr(self, level: int = 0) -> str:
        shared_with_line = ""
        if len(self.output_readers) > 0 and self.output_reader_names is not None:
            owners_string = " and ".join([f"*{x}*" for x in self.output_reader_names])
            shared_with_line += (
                f"Custom Policy: "
                f"outputs are *shared* with the owners of {owners_string} once computed"
            )

        md = f"""class UserCode
    id: UID = {self.id}
    service_func_name: str = {self.service_func_name}
    shareholders: list = {self.input_owners}
    status: list = {self.code_status}
    {shared_with_line}
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
            for _, (obj, _) in self.nested_codes.items():
                code = obj.resolve
                md += "\n"
                md += code._inner_repr(level=level + 1)

        return md

    def _repr_markdown_(self, wrap_as_python: bool = True, indent: int = 0) -> str:
        return as_markdown_code(self._inner_repr())

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


@serializable(without=["local_function"])
class SubmitUserCode(SyftObject):
    # version
    __canonical_name__ = "SubmitUserCode"
    __version__ = SYFT_OBJECT_VERSION_4

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
    enclave_metadata: EnclaveMetadata | None = None
    worker_pool_name: str | None = None

    __repr_attrs__ = ["func_name", "code"]

    @field_validator("output_policy_init_kwargs", mode="after")
    @classmethod
    def add_output_policy_ids(cls, values: Any) -> Any:
        if isinstance(values, dict) and "id" not in values:
            values["id"] = UID()
        return values

    @property
    def kwargs(self) -> dict[Any, Any] | None:
        return self.input_policy_init_kwargs

    def __call__(self, *args: Any, syft_no_node: bool = False, **kwargs: Any) -> Any:
        if syft_no_node:
            return self.local_call(*args, **kwargs)
        return self._ephemeral_node_call(*args, **kwargs)

    def local_call(self, *args: Any, **kwargs: Any) -> Any:
        # only run this on the client side
        if self.local_function:
            tree = ast.parse(inspect.getsource(self.local_function))

            # check there are no globals
            v = GlobalsVisitor()
            v.visit(tree)

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

    def _ephemeral_node_call(
        self,
        time_alive: int | None = None,
        n_consumers: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # relative
        from ... import _orchestra

        # Right now we only create a number of workers
        # In the future we might need to have the same pools/images as well

        if n_consumers is None:
            print(
                SyftInfo(
                    message="Creating a node with n_consumers=2 (the default value)"
                )
            )
            n_consumers = 2

        if time_alive is None and "blocking" in kwargs and not kwargs["blocking"]:
            print(
                SyftInfo(
                    message="Closing the node after time_alive=300 (the default value)"
                )
            )
            time_alive = 300

        # This could be changed given the work on containers
        ep_node = _orchestra().launch(
            name=f"ephemeral_node_{self.func_name}_{random.randint(a=0, b=10000)}",  # nosec
            reset=True,
            create_producer=True,
            n_consumers=n_consumers,
            deploy_to="python",
        )
        ep_client = ep_node.login(email="info@openmined.org", password="changethis")  # nosec
        self.input_policy_init_kwargs = cast(dict, self.input_policy_init_kwargs)
        for node_id, obj_dict in self.input_policy_init_kwargs.items():
            # api = APIRegistry.api_for(
            #     node_uid=node_id.node_id, user_verify_key=node_id.verify_key
            # )
            api = APIRegistry.get_by_recent_node_uid(node_uid=node_id.node_id)
            if api is None:
                return SyftError(
                    f"Can't access the api. You must login to {node_id.node_id}"
                )
            # Creating TwinObject from the ids of the kwargs
            # Maybe there are some corner cases where this is not enough
            # And need only ActionObjects
            # Also, this works only on the assumption that all inputs
            # are ActionObjects, which might change in the future
            for _, id in obj_dict.items():
                mock_obj = api.services.action.get_mock(id)
                if isinstance(mock_obj, SyftError):
                    data_obj = api.services.action.get(id)
                    if isinstance(data_obj, SyftError):
                        return SyftError(
                            message="You do not have access to object you want \
                                to use, or the private object does not have mock \
                                data. Contact the Node Admin."
                        )
                else:
                    data_obj = mock_obj
                data_obj.id = id
                new_obj = ActionObject.from_obj(
                    data_obj.syft_action_data,
                    id=id,
                    syft_node_location=node_id.node_id,
                    syft_client_verify_key=node_id.verify_key,
                )
                res = ep_client.api.services.action.set(new_obj)
                if isinstance(res, SyftError):
                    return res

        new_syft_func = deepcopy(self)

        # This will only be used without worker_pools
        new_syft_func.worker_pool_name = None

        # We will look for subjos, and if we find any will submit them
        # to the ephemeral_node
        submit_subjobs_code(self, ep_client)

        ep_client.code.request_code_execution(new_syft_func)
        ep_client.requests[-1].approve(approve_nested=True)
        func_call = getattr(ep_client.code, new_syft_func.func_name)
        result = func_call(*args, **kwargs)

        def task() -> None:
            if "blocking" in kwargs and not kwargs["blocking"]:
                time.sleep(time_alive)
            print(SyftInfo(message="Landing the ephmeral node..."))
            ep_node.land()
            print(SyftInfo(message="Node Landed!"))

        thread = Thread(target=task)
        thread.start()

        return result

    @property
    def input_owner_verify_keys(self) -> list[str] | None:
        if self.input_policy_init_kwargs is not None:
            return [x.verify_key for x in self.input_policy_init_kwargs.keys()]
        return None


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


def syft_function(
    input_policy: InputPolicy | UID | None = None,
    output_policy: OutputPolicy | UID | None = None,
    share_results_with_owners: bool = False,
    worker_pool_name: str | None = None,
) -> Callable:
    if input_policy is None:
        input_policy = EmpyInputPolicy()

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

    def decorator(f: Any) -> SubmitUserCode:
        res = SubmitUserCode(
            code=inspect.getsource(f),
            func_name=f.__name__,
            signature=inspect.signature(f),
            input_policy_type=input_policy_type,
            input_policy_init_kwargs=getattr(input_policy, "init_kwargs", {}),
            output_policy_type=output_policy_type,
            output_policy_init_kwargs=getattr(output_policy, "init_kwargs", {}),
            local_function=f,
            input_kwargs=f.__code__.co_varnames[: f.__code__.co_argcount],
            worker_pool_name=worker_pool_name,
        )

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


def process_code(
    context: TransformContext,
    raw_code: str,
    func_name: str,
    original_func_name: str,
    policy_input_kwargs: list[str],
    function_input_kwargs: list[str],
) -> str:
    tree = ast.parse(raw_code)

    # check there are no globals
    v = GlobalsVisitor()
    v.visit(tree)

    f = tree.body[0]
    f.decorator_list = []

    call_args = function_input_kwargs
    if "domain" in function_input_kwargs and context.output is not None:
        context.output["uses_domain"] = True
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


def new_check_code(context: TransformContext) -> TransformContext:
    # TODO: remove this tech debt hack
    if context.output is None:
        return context

    input_kwargs = context.output["input_policy_init_kwargs"]
    node_view_workaround = False
    for k in input_kwargs.keys():
        if isinstance(k, NodeIdentity):
            node_view_workaround = True

    if not node_view_workaround:
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
    if context.node is None:
        raise ValueError(f"context {context}'s node is None")
    if context.output is not None:
        nested_codes = {}
        tree = ast.parse(context.output["raw_code"])
        # look for domain arg
        if "domain" in [arg.arg for arg in tree.body[0].args.args]:
            v = LaunchJobVisitor()
            v.visit(tree)
            nested_calls = v.nested_calls
            user_code_service = context.node.get_service("usercodeService")
            for call in nested_calls:
                user_codes = user_code_service.get_by_service_name(context, call)
                if isinstance(user_codes, SyftError):
                    raise Exception(user_codes.message)
                # TODO: Not great
                user_code = user_codes[-1]
                user_code_link = LinkedObject.from_obj(
                    user_code, node_uid=context.node.id
                )
                nested_codes[call] = (user_code_link, user_code.nested_codes)
        context.output["nested_codes"] = nested_codes
    return context


def compile_byte_code(parsed_code: str) -> PyCodeObject | None:
    try:
        return compile(parsed_code, "<string>", "exec")
    except Exception as e:
        print("WARNING: to compile byte code", e)
    return None


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

    code = context.output["code"]
    context.output["raw_code"] = code
    code_hash = hashlib.sha256(code.encode("utf8")).hexdigest()
    context.output["code_hash"] = code_hash

    return context


def add_credentials_for_key(key: str) -> Callable:
    def add_credentials(context: TransformContext) -> TransformContext:
        if context.output is not None:
            context.output[key] = context.credentials
        return context

    return add_credentials


def check_policy(policy: Any, context: TransformContext) -> TransformContext:
    if context.node is not None:
        policy_service = context.node.get_service(PolicyService)
        if isinstance(policy, SubmitUserPolicy):
            policy = policy.to(UserPolicy, context=context)
        elif isinstance(policy, UID):
            policy = policy_service.get_policy_by_uid(context, policy)
            if policy.is_ok():
                policy = policy.ok()
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

    if context.node is None:
        raise ValueError(f"{context}'s node is None")

    if context.output is None:
        return context

    input_keys = list(context.output["input_policy_init_kwargs"].keys())
    code_link = LinkedObject.from_uid(
        context.output["id"],
        UserCode,
        service_type=UserCodeService,
        node_uid=context.node.id,
    )
    if context.node.node_type == NodeType.DOMAIN:
        node_identity = NodeIdentity(
            node_name=context.node.name,
            node_id=context.node.id,
            verify_key=context.node.signing_key.verify_key,
        )
        status = UserCodeStatusCollection(
            status_dict={node_identity: (UserCodeStatus.PENDING, "")},
            user_code_link=code_link,
        )

    elif context.node.node_type == NodeType.ENCLAVE:
        status_dict = {key: (UserCodeStatus.PENDING, "") for key in input_keys}
        status = UserCodeStatusCollection(
            status_dict=status_dict,
            user_code_link=code_link,
        )
    else:
        raise NotImplementedError(
            f"Invalid node type:{context.node.node_type} for code submission"
        )

    res = context.node.get_service("usercodestatusservice").create(context, status)
    # relative
    from .status_service import UserCodeStatusService

    # TODO error handling in transform functions
    if not isinstance(res, SyftError):
        context.output["status_link"] = LinkedObject.from_uid(
            res.id,
            UserCodeStatusCollection,
            service_type=UserCodeStatusService,
            node_uid=context.node.id,
        )
    return context


def add_submit_time(context: TransformContext) -> TransformContext:
    if context.output:
        context.output["submit_time"] = DateTime.now()
    return context


def set_default_pool_if_empty(context: TransformContext) -> TransformContext:
    if (
        context.node
        and context.output
        and context.output.get("worker_pool_name", None) is None
    ):
        default_pool = context.node.get_default_worker_pool()
        context.output["worker_pool_name"] = default_pool.name
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
        add_node_uid_for_key("node_uid"),
        add_submit_time,
        set_default_pool_if_empty,
    ]


@serializable()
class UserCodeExecutionResult(SyftObject):
    # version
    __canonical_name__ = "UserCodeExecutionResult"
    __version__ = SYFT_OBJECT_VERSION_2

    id: UID
    user_code_id: UID
    stdout: str
    stderr: str
    result: Any = None


@serializable()
class UserCodeExecutionOutput(SyftObject):
    # version
    __canonical_name__ = "UserCodeExecutionOutput"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    user_code_id: UID
    stdout: str
    stderr: str
    result: Any = None


class SecureContext:
    def __init__(self, context: AuthedServiceContext) -> None:
        node = context.node
        if node is None:
            raise ValueError(f"{context}'s node is None")

        job_service = node.get_service("jobservice")
        action_service = node.get_service("actionservice")
        # user_service = node.get_service("userservice")

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
        #     user_api = node.get_api(context.credentials, data_protcol.latest_version)
        #     user_api.signing_key = user_signing_key
        #     # We hardcode a python connection here since we have access to the node
        #     # TODO: this is not secure
        #     user_api.connection = PythonConnection(node=node)

        #     APIRegistry.set_api_for(
        #         node_uid=node.id,
        #         user_verify_key=context.credentials,
        #         api=user_api,
        #     )

        def launch_job(func: UserCode, **kwargs: Any) -> Job | None:
            # relative

            kw2id = {}
            for k, v in kwargs.items():
                value = ActionObject.from_obj(v)
                ptr = action_service._set(context, value)
                ptr = ptr.ok()
                kw2id[k] = ptr.id
            try:
                # TODO: check permissions here
                action = Action.syft_function_action_from_kwargs_and_id(kw2id, func.id)

                job = node.add_action_to_queue(
                    action=action,
                    credentials=context.credentials,
                    parent_job_id=context.job_id,
                    has_execute_permissions=True,
                    worker_pool_name=func.worker_pool_name,
                )
                # # set api in global scope to enable using .get(), .wait())
                # set_api_registry()

                return job
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

        class LocalDomainClient:
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
                if context.node is not None:
                    log_service = context.node.get_service("LogService")
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

        if code_item.uses_domain:
            kwargs["domain"] = LocalDomainClient()

        for k, v in kwargs.items():
            if isinstance(v, CustomEndpointActionObject):
                kwargs[k] = v.add_context(context=context)

        stdout = StringIO()
        stderr = StringIO()

        # statisfy lint checker
        result = None

        # We only need access to local kwargs
        _locals = {"kwargs": kwargs}
        _globals = {}
        if code_item.nested_codes is not None:
            for service_func_name, (linked_obj, _) in code_item.nested_codes.items():
                code_obj = linked_obj.resolve_with_context(context=context)
                if isinstance(code_obj, Err):
                    raise Exception(code_obj.err())
                _globals[service_func_name] = code_obj.ok()
        _globals["print"] = print
        exec(code_item.parsed_code, _globals, _locals)  # nosec

        evil_string = f"{code_item.unique_func_name}(**kwargs)"
        try:
            result = eval(evil_string, _globals, _locals)  # nosec
        except Exception as e:
            error_msg = traceback_from_error(e, code_item)
            if context.job is not None:
                time = datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")
                original_print(
                    f"{time} EXCEPTION LOG ({job_id}):\n{error_msg}", file=sys.stderr
                )
            if context.node is not None:
                log_id = context.job.log_id
                log_service = context.node.get_service("LogService")
                log_service.append(context=context, uid=log_id, new_err=error_msg)

            result_message = (
                f"Exception encountered while running {code_item.service_func_name}"
                ", please contact the Node Admin for more info."
            )
            if context.dev_mode:
                result_message += error_msg

            result = Err(result_message)

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
        )

    except Exception as e:
        # stdlib

        print = original_print
        # print("execute_byte_code failed", e, file=stderr_)
        print(traceback.format_exc())
        print("execute_byte_code failed", e)
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
    try:
        for user_code in user_code_items:
            if context is None:
                status = user_code.status
            else:
                status = user_code.get_status(context)

            if status.approved:
                if isinstance(user_code.input_policy_type, UserPolicy):
                    load_policy_code(user_code.input_policy_type)
                if isinstance(user_code.output_policy_type, UserPolicy):
                    load_policy_code(user_code.output_policy_type)
    except Exception as e:
        raise Exception(f"Failed to load code: {user_code}: {e}")
