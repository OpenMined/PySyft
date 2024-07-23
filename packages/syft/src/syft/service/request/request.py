# stdlib
from collections.abc import Callable
from enum import Enum
import hashlib
import inspect
import logging
from typing import Any

# third party
from pydantic import model_validator
from result import Err
from result import Ok
from result import Result
from typing_extensions import Self

# relative
from ...abstract_server import ServerSideType
from ...client.api import APIRegistry
from ...client.client import SyftClient
from ...custom_worker.config import DockerWorkerConfig
from ...custom_worker.config import WorkerConfig
from ...custom_worker.k8s import IN_KUBERNETES
from ...serde.serializable import serializable
from ...serde.serialize import _serialize
from ...server.credentials import SyftVerifyKey
from ...store.linked_obj import LinkedObject
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.syncable_object import SyncableSyftObject
from ...types.transforms import TransformContext
from ...types.transforms import add_server_uid_for_key
from ...types.transforms import generate_id
from ...types.transforms import transform
from ...types.twin_object import TwinObject
from ...types.uid import LineageID
from ...types.uid import UID
from ...util import options
from ...util.colors import SURFACE
from ...util.decorators import deprecated
from ...util.markdown import markdown_as_class_with_fields
from ...util.notebook_ui.icons import Icon
from ...util.util import prompt_warning_message
from ..action.action_object import ActionObject
from ..action.action_service import ActionService
from ..action.action_store import ActionObjectPermission
from ..action.action_store import ActionPermission
from ..blob_storage.service import BlobStorageService
from ..code.user_code import UserCode
from ..code.user_code import UserCodeStatus
from ..code.user_code import UserCodeStatusCollection
from ..context import AuthedServiceContext
from ..context import ChangeContext
from ..job.job_stash import Job
from ..job.job_stash import JobStatus
from ..notification.notifications import Notification
from ..response import SyftError
from ..response import SyftSuccess
from ..user.user import UserView

logger = logging.getLogger(__name__)


@serializable(canonical_name="RequestStatus", version=1)
class RequestStatus(Enum):
    PENDING = 0
    REJECTED = 1
    APPROVED = 2

    @classmethod
    def from_usercode_status(cls, status: UserCodeStatusCollection) -> "RequestStatus":
        if status.approved:
            return RequestStatus.APPROVED
        elif status.denied:
            return RequestStatus.REJECTED
        else:
            return RequestStatus.PENDING


@serializable()
class Change(SyftObject):
    __canonical_name__ = "Change"
    __version__ = SYFT_OBJECT_VERSION_1

    linked_obj: LinkedObject | None = None

    def change_object_is_type(self, type_: type) -> bool:
        return self.linked_obj is not None and type_ == self.linked_obj.object_type


@serializable()
class ChangeStatus(SyftObject):
    __canonical_name__ = "ChangeStatus"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID | None = None  # type: ignore[assignment]
    change_id: UID
    applied: bool = False

    @classmethod
    def from_change(cls, change: Change, applied: bool) -> Self:
        return cls(change_id=change.id, applied=applied)


@serializable()
class ActionStoreChange(Change):
    __canonical_name__ = "ActionStoreChange"
    __version__ = SYFT_OBJECT_VERSION_1

    linked_obj: LinkedObject
    apply_permission_type: ActionPermission

    __repr_attrs__ = ["linked_obj", "apply_permission_type"]

    def _run(
        self, context: ChangeContext, apply: bool
    ) -> Result[SyftSuccess, SyftError]:
        try:
            action_service: ActionService = context.server.get_service(ActionService)  # type: ignore[assignment]
            blob_storage_service = context.server.get_service(BlobStorageService)
            action_store = action_service.store

            # can we ever have a lineage ID in the store?
            obj_uid = self.linked_obj.object_uid
            obj_uid = obj_uid.id if isinstance(obj_uid, LineageID) else obj_uid

            action_obj = action_store.get(
                uid=obj_uid,
                credentials=context.approving_user_credentials,
            )

            if action_obj.is_err():
                return Err(SyftError(message=f"{action_obj.err()}"))

            action_obj = action_obj.ok()

            owner_permission = ActionObjectPermission(
                uid=obj_uid,
                credentials=context.approving_user_credentials,
                permission=self.apply_permission_type,
            )
            if action_store.has_permission(permission=owner_permission):
                id_action = (
                    action_obj.id
                    if not isinstance(action_obj.id, LineageID)
                    else action_obj.id.id
                )
                requesting_permission_action_obj = ActionObjectPermission(
                    uid=id_action,
                    credentials=context.requesting_user_credentials,
                    permission=self.apply_permission_type,
                )
                if isinstance(action_obj, TwinObject):
                    uid_blob = action_obj.private.syft_blob_storage_entry_id
                else:
                    uid_blob = action_obj.syft_blob_storage_entry_id
                requesting_permission_blob_obj = (
                    ActionObjectPermission(
                        uid=uid_blob,
                        credentials=context.requesting_user_credentials,
                        permission=self.apply_permission_type,
                    )
                    if uid_blob
                    else None
                )
                if apply:
                    logger.debug(
                        "ADDING PERMISSION", requesting_permission_action_obj, id_action
                    )
                    action_store.add_permission(requesting_permission_action_obj)
                    (
                        blob_storage_service.stash.add_permission(
                            requesting_permission_blob_obj
                        )
                        if requesting_permission_blob_obj
                        else None
                    )
                else:
                    if action_store.has_permission(requesting_permission_action_obj):
                        action_store.remove_permission(requesting_permission_action_obj)
                    if (
                        requesting_permission_blob_obj
                        and blob_storage_service.stash.has_permission(
                            requesting_permission_blob_obj
                        )
                    ):
                        blob_storage_service.stash.remove_permission(
                            requesting_permission_blob_obj
                        )
            else:
                return Err(
                    SyftError(
                        message=f"No permission for approving_user_credentials {context.approving_user_credentials}"
                    )
                )
            return Ok(SyftSuccess(message=f"{type(self)} Success"))
        except Exception as e:
            logger.error(f"failed to apply {type(self)}", exc_info=e)
            return Err(SyftError(message=str(e)))

    def apply(self, context: ChangeContext) -> Result[SyftSuccess, SyftError]:
        return self._run(context=context, apply=True)

    def undo(self, context: ChangeContext) -> Result[SyftSuccess, SyftError]:
        return self._run(context=context, apply=False)

    def __repr_syft_nested__(self) -> str:
        return f"Apply <b>{self.apply_permission_type}</b> to \
<i>{self.linked_obj.object_type.__canonical_name__}:{self.linked_obj.object_uid.short()}</i>."


@serializable()
class CreateCustomImageChange(Change):
    __canonical_name__ = "CreateCustomImageChange"
    __version__ = SYFT_OBJECT_VERSION_1

    config: WorkerConfig
    tag: str | None = None
    registry_uid: UID | None = None
    pull_image: bool = True

    __repr_attrs__ = ["config", "tag"]

    @model_validator(mode="after")
    def _tag_required_for_dockerworkerconfig(self) -> Self:
        if isinstance(self.config, DockerWorkerConfig) and self.tag is None:
            raise ValueError("`tag` is required for `DockerWorkerConfig`.")
        return self

    def _run(
        self, context: ChangeContext, apply: bool
    ) -> Result[SyftSuccess, SyftError]:
        try:
            worker_image_service = context.server.get_service("SyftWorkerImageService")

            service_context = context.to_service_ctx()
            result = worker_image_service.submit(
                service_context, worker_config=self.config
            )

            if isinstance(result, SyftError):
                return Err(result)

            result = worker_image_service.stash.get_by_worker_config(
                service_context.credentials, config=self.config
            )

            if result.is_err():
                return Err(SyftError(message=f"{result.err()}"))

            if (worker_image := result.ok()) is None:
                return Err(SyftError(message="The worker image does not exist."))

            build_success_message = "Image was pre-built."

            if not worker_image.is_prebuilt:
                build_result = worker_image_service.build(
                    service_context,
                    image_uid=worker_image.id,
                    tag=self.tag,
                    registry_uid=self.registry_uid,
                    pull_image=self.pull_image,
                )

                if isinstance(build_result, SyftError):
                    return Err(build_result)

                build_success_message = build_result.message

            build_success = SyftSuccess(
                message=f"Build result: {build_success_message}"
            )

            if IN_KUBERNETES and not worker_image.is_prebuilt:
                push_result = worker_image_service.push(
                    service_context,
                    image_uid=worker_image.id,
                    username=context.extra_kwargs.get("registry_username", None),
                    password=context.extra_kwargs.get("registry_password", None),
                )

                if isinstance(push_result, SyftError):
                    return Err(push_result)

                return Ok(
                    SyftSuccess(
                        message=f"{build_success}\nPush result: {push_result.message}"
                    )
                )

            return Ok(build_success)

        except Exception as e:
            return Err(SyftError(message=f"Failed to create/build image: {e}"))

    def apply(self, context: ChangeContext) -> Result[SyftSuccess, SyftError]:
        return self._run(context=context, apply=True)

    def undo(self, context: ChangeContext) -> Result[SyftSuccess, SyftError]:
        return self._run(context=context, apply=False)

    def __repr_syft_nested__(self) -> str:
        return f"Create Image for Config: {self.config} with tag: {self.tag}"


@serializable()
class CreateCustomWorkerPoolChange(Change):
    __canonical_name__ = "CreateCustomWorkerPoolChange"
    __version__ = SYFT_OBJECT_VERSION_1

    pool_name: str
    num_workers: int
    image_uid: UID | None = None
    config: WorkerConfig | None = None
    pod_annotations: dict[str, str] | None = None
    pod_labels: dict[str, str] | None = None

    __repr_attrs__ = ["pool_name", "num_workers", "image_uid"]

    def _run(
        self, context: ChangeContext, apply: bool
    ) -> Result[SyftSuccess, SyftError]:
        """
        This function is run when the DO approves (apply=True)
        or deny (apply=False) the request.
        """
        # TODO: refactor the returned Err(SyftError) or Ok(SyftSuccess) to just
        # SyftError or SyftSuccess
        if apply:
            # get the worker pool service and try to launch a pool
            worker_pool_service = context.server.get_service("SyftWorkerPoolService")
            service_context: AuthedServiceContext = context.to_service_ctx()

            if self.config is not None:
                result = worker_pool_service.image_stash.get_by_worker_config(
                    service_context.credentials, self.config
                )
                if result.is_err():
                    return Err(SyftError(message=f"{result.err()}"))
                worker_image = result.ok()
                self.image_uid = worker_image.id

            result = worker_pool_service.launch(
                context=service_context,
                pool_name=self.pool_name,
                image_uid=self.image_uid,
                num_workers=self.num_workers,
                registry_username=context.extra_kwargs.get("registry_username", None),
                registry_password=context.extra_kwargs.get("registry_password", None),
                pod_annotations=self.pod_annotations,
                pod_labels=self.pod_labels,
            )
            if isinstance(result, SyftError):
                return Err(result)
            else:
                return Ok(result)
        else:
            return Err(
                SyftError(
                    message=f"Request to create a worker pool with name {self.name} denied"
                )
            )

    def apply(self, context: ChangeContext) -> Result[SyftSuccess, SyftError]:
        return self._run(context=context, apply=True)

    def undo(self, context: ChangeContext) -> Result[SyftSuccess, SyftError]:
        return self._run(context=context, apply=False)

    def __repr_syft_nested__(self) -> str:
        return (
            f"Create Worker Pool '{self.pool_name}' for Image with id {self.image_uid}"
        )


@serializable()
class Request(SyncableSyftObject):
    __canonical_name__ = "Request"
    __version__ = SYFT_OBJECT_VERSION_1

    requesting_user_verify_key: SyftVerifyKey
    requesting_user_name: str = ""
    requesting_user_email: str | None = ""
    requesting_user_institution: str | None = ""
    approving_user_verify_key: SyftVerifyKey | None = None
    request_time: DateTime
    updated_at: DateTime | None = None
    server_uid: UID
    request_hash: str
    changes: list[Change]
    history: list[ChangeStatus] = []
    tags: list[str] = []

    __table_coll_widths__ = [
        "min-content",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
    ]

    __attr_searchable__ = [
        "requesting_user_verify_key",
        "approving_user_verify_key",
    ]
    __attr_unique__ = ["request_hash"]
    __repr_attrs__ = [
        "request_time",
        "updated_at",
        "status",
        "changes",
        "requesting_user_verify_key",
    ]
    __exclude_sync_diff_attrs__ = ["server_uid", "changes", "history"]
    __table_sort_attr__ = "Request time"

    def _repr_html_(self) -> Any:
        # add changes
        updated_at_line = ""
        if self.updated_at is not None:
            updated_at_line += (
                f"<p><strong>Created by: </strong>{self.requesting_user_name}</p>"
            )
        str_changes_ = []
        for change in self.changes:
            str_change = (
                change.__repr_syft_nested__()
                if hasattr(change, "__repr_syft_nested__")
                else type(change)
            )
            str_change = f"{str_change}. "
            str_changes_.append(str_change)
        str_changes = "\n".join(str_changes_)
        api = APIRegistry.api_for(
            self.server_uid,
            self.syft_client_verify_key,
        )
        shared_with_line = ""
        if self.code and len(self.code.output_readers) > 0:
            # owner_names = ["canada", "US"]
            owners_string = " and ".join(
                [f"<strong>{x}</strong>" for x in self.code.output_reader_names]  # type: ignore
            )
            shared_with_line += (
                f"<p><strong>Custom Policy: </strong> "
                f"outputs are <strong>shared</strong> with the owners of {owners_string} once computed"
            )

        server_info = ""
        if api is not None:
            metadata = api.services.metadata.get_metadata()
            server_name = (
                api.server_name.capitalize() if api.server_name is not None else ""
            )
            server_type = metadata.server_type.value.capitalize()
            server_info = f"<p><strong>Requested on: </strong> {server_name} of type <strong>{server_type}</strong></p>"

        email_str = (
            f"({self.requesting_user_email})" if self.requesting_user_email else ""
        )
        institution_str = (
            f"<strong>Institution:</strong> {self.requesting_user_institution}"
            if self.requesting_user_institution
            else ""
        )

        return f"""
            <style>
            .syft-request {{color: {SURFACE[options.color_theme]};}}
            </style>
            <div class='syft-request'>
                <h3>Request</h3>
                <p><strong>Id: </strong>{self.id}</p>
                <p><strong>Request time: </strong>{self.request_time}</p>
                {updated_at_line}
                {shared_with_line}
                <p><strong>Status: </strong>{self.status}</p>
                {server_info}
                <p><strong>Requested by:</strong> {self.requesting_user_name} {email_str} {institution_str}</p>
                <p><strong>Changes: </strong> {str_changes}</p>
            </div>

            """

    @property
    def html_description(self) -> str:
        desc = " ".join([x.__repr_syft_nested__() for x in self.changes])
        # desc = desc.replace('\n', '')
        # desc = desc.replace('<br>', '\n')
        desc = desc.replace(". ", ".\n\n")
        desc = desc.replace("<b>", "")
        desc = desc.replace("</b>", "")
        desc = desc.replace("<i>", "")
        desc = desc.replace("</i>", "")

        return desc

    def _coll_repr_(self) -> dict[str, str | dict[str, str]]:
        if self.status == RequestStatus.APPROVED:
            badge_color = "badge-green"
        elif self.status == RequestStatus.PENDING:
            badge_color = "badge-gray"
        else:
            badge_color = "badge-red"

        status_badge = {"value": self.status.name.capitalize(), "type": badge_color}

        user_data = [
            self.requesting_user_name,
            self.requesting_user_email,
            self.requesting_user_institution,
        ]

        return {
            "Description": self.html_description,
            "Requested By": "\n".join(user_data),
            "Creation Time": str(self.request_time),
            "Status": status_badge,
        }

    @property
    def code_id(self) -> UID:
        for change in self.changes:
            if isinstance(change, UserCodeStatusChange):
                return change.linked_user_code.object_uid
        return SyftError(
            message="This type of request does not have code associated with it."
        )

    @property
    def codes(self) -> Any:
        for change in self.changes:
            if isinstance(change, UserCodeStatusChange):
                return change.codes
        return SyftError(
            message="This type of request does not have code associated with it."
        )

    def get_user_code(self, context: AuthedServiceContext) -> UserCode | None:
        for change in self.changes:
            if isinstance(change, UserCodeStatusChange):
                return change.get_user_code(context)
        return None

    @property
    def code(self) -> UserCode | SyftError:
        for change in self.changes:
            if isinstance(change, UserCodeStatusChange):
                return change.code
        return SyftError(
            message="This type of request does not have code associated with it."
        )

    @property
    def current_change_state(self) -> dict[UID, bool]:
        change_applied_map = {}
        for change_status in self.history:
            # only store the last change
            change_applied_map[change_status.change_id] = change_status.applied

        return change_applied_map

    @property
    def icon(self) -> str:
        return Icon.REQUEST.svg

    def get_status(self, context: AuthedServiceContext | None = None) -> RequestStatus:
        is_l0_deployment = (
            self.get_is_l0_deployment(context) if context else self.is_l0_deployment
        )
        if is_l0_deployment:
            code_status = self.code.get_status(context) if context else self.code.status
            return RequestStatus.from_usercode_status(code_status)

        if len(self.history) == 0:
            return RequestStatus.PENDING

        all_changes_applied = all(self.current_change_state.values()) and (
            len(self.current_change_state) == len(self.changes)
        )

        request_status = (
            RequestStatus.APPROVED if all_changes_applied else RequestStatus.REJECTED
        )

        return request_status

    @property
    def status(self) -> RequestStatus:
        return self.get_status()

    def approve(
        self,
        disable_warnings: bool = False,
        approve_nested: bool = False,
        **kwargs: dict,
    ) -> Result[SyftSuccess, SyftError]:
        api = self._get_api()
        if isinstance(api, SyftError):
            return api

        if self.is_l0_deployment:
            return SyftError(
                message="This request is a low-side request. Please sync your results to approve."
            )
        # TODO: Refactor so that object can also be passed to generate warnings
        if api.connection:
            metadata = api.connection.get_server_metadata(api.signing_key)
        else:
            metadata = None
        message = None

        is_code_request = not isinstance(self.codes, SyftError)

        if is_code_request and len(self.codes) > 1 and not approve_nested:
            return SyftError(
                message="Multiple codes detected, please use approve_nested=True"
            )

        if metadata and metadata.server_side_type == ServerSideType.HIGH_SIDE.value:
            message = (
                "You're approving a request on "
                f"{metadata.server_side_type} side {metadata.server_type} "
                "which may host datasets with private information."
            )
        if message and metadata and metadata.show_warnings and not disable_warnings:
            prompt_warning_message(message=message, confirm=True)
        msg = (
            "Approving request ",
            f"on change {self.code.service_func_name} " if is_code_request else "",
            f"for datasite {api.server_name}",
        )

        print("".join(msg))
        res = api.services.request.apply(self.id, **kwargs)
        return res

    def deny(self, reason: str) -> SyftSuccess | SyftError:
        """Denies the particular request.

        Args:
            reason (str): Reason for which the request has been denied.
        """
        api = self._get_api()
        if isinstance(api, SyftError):
            return api

        if self.is_l0_deployment:
            if self.status == RequestStatus.APPROVED:
                prompt_warning_message(
                    "This request already has results published to the data scientist. "
                    "They will still be able to access those results."
                )
            result = api.code.update(id=self.code_id, l0_deny_reason=reason)
            if isinstance(result, SyftError):
                return result
            return SyftSuccess(message=f"Request denied with reason: {reason}")

        return api.services.request.undo(uid=self.id, reason=reason)

    @property
    def is_l0_deployment(self) -> bool:
        return bool(self.code) and self.code.is_l0_deployment

    def get_is_l0_deployment(self, context: AuthedServiceContext) -> bool:
        code = self.get_user_code(context)
        if code:
            return code.is_l0_deployment
        else:
            return False

    def approve_with_client(self, client: SyftClient) -> Result[SyftSuccess, SyftError]:
        if self.is_l0_deployment:
            return SyftError(
                message="This request is a low-side request. Please sync your results to approve."
            )

        print(f"Approving request for datasite {client.name}")
        return client.api.services.request.apply(self.id)

    def apply(self, context: AuthedServiceContext) -> Result[SyftSuccess, SyftError]:
        change_context: ChangeContext = ChangeContext.from_service(context)
        change_context.requesting_user_credentials = self.requesting_user_verify_key
        for change in self.changes:
            # by default change status is not applied
            change_status = ChangeStatus(change_id=change.id, applied=False)
            result = change.apply(context=change_context)
            if isinstance(result, SyftError):
                return result
            if result.is_err():
                # add to history and save history to request
                self.history.append(change_status)
                self.save(context=context)
                return result

            # If no error, then change successfully applied.
            change_status.applied = True
            self.history.append(change_status)

        self.updated_at = DateTime.now()
        self.save(context=context)

        return Ok(SyftSuccess(message=f"Request {self.id} changes applied"))

    def undo(self, context: AuthedServiceContext) -> Result[SyftSuccess, SyftError]:
        change_context: ChangeContext = ChangeContext.from_service(context)
        change_context.requesting_user_credentials = self.requesting_user_verify_key

        current_change_state = self.current_change_state
        for change in self.changes:
            # by default change status is not applied
            is_change_applied = current_change_state.get(change.id, False)
            change_status = ChangeStatus(
                change_id=change.id,
                applied=is_change_applied,
            )
            # undo here may be deny for certain Changes (UserCodeChange)
            result = change.undo(context=change_context)
            if result.is_err():
                # add to history and save history to request
                self.history.append(change_status)
                self.save(context=context)
                return result

            # If no error, then change successfully undone.
            change_status.applied = False
            self.history.append(change_status)

        self.updated_at = DateTime.now()
        result = self.save(context=context)
        if isinstance(result, SyftError):
            return Err(result)
        # override object with latest changes.
        self = result
        return Ok(SyftSuccess(message=f"Request {self.id} changes undone."))

    def save(self, context: AuthedServiceContext) -> Result[SyftSuccess, SyftError]:
        # relative
        from .request_service import RequestService

        save_method = context.server.get_service_method(RequestService.save)
        return save_method(context=context, request=self)

    def _create_action_object_for_deposited_result(
        self,
        result: Any,
    ) -> ActionObject | SyftError:
        api = self._get_api()
        if isinstance(api, SyftError):
            return api

        # Ensure result is an ActionObject
        if isinstance(result, ActionObject):
            existing_job = api.services.job.get_by_result_id(result.id.id)
            if existing_job is not None:
                return SyftError(
                    message=f"This ActionObject is already the result of Job {existing_job.id}"
                )
            action_object = result
        else:
            action_object = ActionObject.from_obj(
                result,
                syft_client_verify_key=self.syft_client_verify_key,
                syft_server_location=self.syft_server_location,
            )

        # Ensure ActionObject exists on this server
        action_object_is_from_this_server = isinstance(
            api.services.action.exists(action_object.id.id), SyftSuccess
        )
        if (
            action_object.syft_blob_storage_entry_id is None
            or not action_object_is_from_this_server
        ):
            action_object.reload_cache()
            result = action_object._send(self.server_uid, self.syft_client_verify_key)
            if isinstance(result, SyftError):
                return result

        return action_object

    def _create_output_history_for_deposited_result(
        self, job: Job, result: Any
    ) -> SyftSuccess | SyftError:
        code = self.code
        if isinstance(code, SyftError):
            return code
        api = self._get_api()
        if isinstance(api, SyftError):
            return api

        input_ids = {}
        input_policy = code.input_policy
        if input_policy is not None:
            for input_ in input_policy.inputs.values():
                input_ids.update(input_)
        res = api.services.code.store_execution_output(
            user_code_id=code.id,
            outputs=result,
            job_id=job.id,
            input_ids=input_ids,
        )

        return res

    def deposit_result(
        self,
        result: Any,
        log_stdout: str = "",
        log_stderr: str = "",
    ) -> Job | SyftError:
        """
        Adds a result to this Request:
        - Create an ActionObject from the result (if not already an ActionObject)
        - Ensure ActionObject exists on this server
        - Create Job with new result and logs
        - Update the output history

        Args:
            result (Any): ActionObject or any object to be saved as an ActionObject.
            logs (str | None, optional): Optional logs to be saved with the Job. Defaults to None.

        Returns:
            Job | SyftError: Job object if successful, else SyftError.
        """

        # TODO check if this is a low-side request. If not, SyftError

        api = self._get_api()
        if isinstance(api, SyftError):
            return api
        code = self.code
        if isinstance(code, SyftError):
            return code

        if not self.is_l0_deployment:
            return SyftError(
                message="deposit_result is only available for low side code requests. "
                "Please use request.approve() instead."
            )

        # Create ActionObject
        action_object = self._create_action_object_for_deposited_result(result)
        if isinstance(action_object, SyftError):
            return action_object

        # Create Job
        # NOTE code owner read permissions are added when syncing this Job
        job = api.services.job.create_job_for_user_code_id(
            code.id,
            result=action_object,
            log_stdout=log_stdout,
            log_stderr=log_stderr,
            status=JobStatus.COMPLETED,
            add_code_owner_read_permissions=False,
        )
        if isinstance(job, SyftError):
            return job

        # Add to output history
        res = self._create_output_history_for_deposited_result(job, result)
        if isinstance(res, SyftError):
            return res

        return job

    @deprecated(
        return_syfterror=True,
        reason="accept_by_depositing_result has been removed. Use approve instead to "
        "approve this request, or deposit_result to deposit a new result.",
    )
    def accept_by_depositing_result(self, result: Any, force: bool = False) -> Any:
        pass

    def get_sync_dependencies(
        self, context: AuthedServiceContext
    ) -> list[UID] | SyftError:
        dependencies = []
        code_id = self.code_id
        if isinstance(code_id, SyftError):
            return code_id
        dependencies.append(code_id)

        return dependencies


@serializable()
class RequestInfo(SyftObject):
    # version
    __canonical_name__ = "RequestInfo"
    __version__ = SYFT_OBJECT_VERSION_1

    user: UserView
    request: Request
    message: Notification


@serializable()
class RequestInfoFilter(SyftObject):
    # version
    __canonical_name__ = "RequestInfoFilter"
    __version__ = SYFT_OBJECT_VERSION_1

    name: str | None = None


@serializable()
class SubmitRequest(SyftObject):
    __canonical_name__ = "SubmitRequest"
    __version__ = SYFT_OBJECT_VERSION_1

    changes: list[Change]
    requesting_user_verify_key: SyftVerifyKey | None = None


def hash_changes(context: TransformContext) -> TransformContext:
    if context.output is None:
        return context

    request_time = context.output["request_time"]
    key = context.output["requesting_user_verify_key"]
    changes = context.output["changes"]

    time_hash = hashlib.sha256(
        _serialize(request_time.utc_timestamp, to_bytes=True)
    ).digest()
    key_hash = hashlib.sha256(bytes(key.verify_key)).digest()
    changes_hash = hashlib.sha256(_serialize(changes, to_bytes=True)).digest()
    final_hash = hashlib.sha256(time_hash + key_hash + changes_hash).hexdigest()

    context.output["request_hash"] = final_hash

    return context


def add_request_time(context: TransformContext) -> TransformContext:
    if context.output is not None:
        context.output["request_time"] = DateTime.now()
    return context


def check_requesting_user_verify_key(context: TransformContext) -> TransformContext:
    if context.output and context.server and context.obj:
        if context.obj.requesting_user_verify_key and context.server.is_root(
            context.credentials
        ):
            context.output["requesting_user_verify_key"] = (
                context.obj.requesting_user_verify_key
            )
        else:
            context.output["requesting_user_verify_key"] = context.credentials

    return context


def add_requesting_user_info(context: TransformContext) -> TransformContext:
    if context.output is not None and context.server is not None:
        try:
            user_key = context.output["requesting_user_verify_key"]
            user_service = context.server.get_service("UserService")
            user = user_service.get_by_verify_key(user_key)
            context.output["requesting_user_name"] = user.name
            context.output["requesting_user_email"] = user.email
            context.output["requesting_user_institution"] = (
                user.institution if user.institution else ""
            )
        except Exception:
            context.output["requesting_user_name"] = "guest_user"

    return context


@transform(SubmitRequest, Request)
def submit_request_to_request() -> list[Callable]:
    return [
        generate_id,
        add_server_uid_for_key("server_uid"),
        add_request_time,
        check_requesting_user_verify_key,
        add_requesting_user_info,
        hash_changes,
    ]


@serializable()
class ObjectMutation(Change):
    __canonical_name__ = "ObjectMutation"
    __version__ = SYFT_OBJECT_VERSION_1

    linked_obj: LinkedObject | None = None
    attr_name: str
    value: Any | None = None
    match_type: bool
    previous_value: Any | None = None

    __repr_attrs__ = ["linked_obj", "attr_name"]

    def mutate(self, obj: Any, value: Any | None = None) -> Any:
        # check if attribute is a property setter first
        # this seems necessary for pydantic types
        attr = getattr(type(obj), self.attr_name, None)
        if inspect.isdatadescriptor(attr):
            if hasattr(attr, "fget") and hasattr(attr, "fset"):
                self.previous_value = attr.fget(obj)
                attr.fset(obj, value)
        else:
            self.previous_value = getattr(obj, self.attr_name, None)
            setattr(obj, self.attr_name, value)
        return obj

    def __repr_syft_nested__(self) -> str:
        return f"Mutate <b>{self.attr_name}</b> to <b>{self.value}</b>"

    def _run(
        self, context: ChangeContext, apply: bool
    ) -> Result[SyftSuccess, SyftError]:
        if self.linked_obj is None:
            return Err(SyftError(message=f"{self}'s linked object is None"))
        try:
            obj = self.linked_obj.resolve_with_context(context)
            if obj.is_err():
                return Err(SyftError(message=obj.err()))
            obj = obj.ok()
            if apply:
                obj = self.mutate(obj, value=self.value)
                self.linked_obj.update_with_context(context, obj)
            else:
                # unset the set value
                obj = self.mutate(obj, value=self.previous_value)
                self.linked_obj.update_with_context(context, obj)

            return Ok(SyftSuccess(message=f"{type(self)} Success"))
        except Exception as e:
            print(f"failed to apply {type(self)}. {e}")
            return Err(SyftError(message=str(e)))

    def apply(self, context: ChangeContext) -> Result[SyftSuccess, SyftError]:
        return self._run(context=context, apply=True)

    def undo(self, context: ChangeContext) -> Result[SyftSuccess, SyftError]:
        return self._run(context=context, apply=False)


def type_for_field(object_type: type, attr_name: str) -> type | None:
    field_type = None
    try:
        field_type = object_type.__dict__["__annotations__"][attr_name]
    except Exception:  # nosec
        try:
            field_type = object_type.__fields__.get(attr_name, None).type_
        except Exception:  # nosec
            pass
    return field_type


@serializable()
class EnumMutation(ObjectMutation):
    __canonical_name__ = "EnumMutation"
    __version__ = SYFT_OBJECT_VERSION_1

    enum_type: type[Enum]
    value: Enum | None = None
    match_type: bool = True

    __repr_attrs__ = ["linked_obj", "attr_name", "value"]

    @property
    def valid(self) -> SyftSuccess | SyftError:
        if self.match_type and not isinstance(self.value, self.enum_type):
            return SyftError(
                message=f"{type(self.value)} must be of type: {self.enum_type}"
            )
        return SyftSuccess(message=f"{type(self)} valid")

    @staticmethod
    def from_obj(
        linked_obj: LinkedObject, attr_name: str, value: Enum | None = None
    ) -> "EnumMutation":
        enum_type = type_for_field(linked_obj.object_type, attr_name)
        return EnumMutation(
            linked_obj=linked_obj,
            attr_name=attr_name,
            enum_type=enum_type,
            value=value,
            match_type=True,
        )

    def _run(
        self, context: ChangeContext, apply: bool
    ) -> Result[SyftSuccess, SyftError]:
        try:
            valid = self.valid
            if not valid:
                return Err(valid)
            if self.linked_obj is None:
                return Err(SyftError(message=f"{self}'s linked object is None"))
            obj = self.linked_obj.resolve_with_context(context)
            if obj.is_err():
                return Err(SyftError(message=obj.err()))
            obj = obj.ok()
            if apply:
                obj = self.mutate(obj=obj)

                self.linked_obj.update_with_context(context, obj)
            else:
                raise NotImplementedError
            return Ok(SyftSuccess(message=f"{type(self)} Success"))
        except Exception as e:
            print(f"failed to apply {type(self)}. {e}")
            return Err(SyftError(message=e))

    def apply(self, context: ChangeContext) -> Result[SyftSuccess, SyftError]:
        return self._run(context=context, apply=True)

    def undo(self, context: ChangeContext) -> Result[SyftSuccess, SyftError]:
        return self._run(context=context, apply=False)

    def __repr_syft_nested__(self) -> str:
        return f"Mutate <b>{self.enum_type}</b> to <b>{self.value}</b>"

    @property
    def link(self) -> SyftObject | None:
        if self.linked_obj:
            return self.linked_obj.resolve
        return None


@serializable()
class UserCodeStatusChange(Change):
    __canonical_name__ = "UserCodeStatusChange"
    __version__ = SYFT_OBJECT_VERSION_1

    value: UserCodeStatus
    linked_obj: LinkedObject
    linked_user_code: LinkedObject
    nested_solved: bool = False
    match_type: bool = True
    __repr_attrs__ = [
        "code.service_func_name",
        "code.input_policy_type.__canonical_name__",
        "code.output_policy_type.__canonical_name__",
        "code.worker_pool_name",
        "code.status.approved",
    ]

    @property
    def code(self) -> UserCode:
        if self.linked_user_code._resolve_cache:
            return self.linked_user_code._resolve_cache
        return self.linked_user_code.resolve

    def get_user_code(self, context: AuthedServiceContext) -> UserCode:
        resolve = self.linked_user_code.resolve_with_context(context)
        return resolve.ok()

    @property
    def codes(self) -> list[UserCode]:
        def recursive_code(server: Any) -> list:
            codes = []
            for obj, new_server in server.values():
                codes.append(obj.resolve)
                codes.extend(recursive_code(new_server))
            return codes

        codes = [self.code]
        codes.extend(recursive_code(self.code.nested_codes))
        return codes

    def nested_repr(self, server: Any | None = None, level: int = 0) -> str:
        msg = ""
        if server is None:
            server = self.code.nested_codes

        for service_func_name, (_, new_server) in server.items():  # type: ignore
            msg = "├──" + "──" * level + f"{service_func_name}<br>"
            msg += self.nested_repr(server=new_server, level=level + 1)
        return msg

    def __repr_syft_nested__(self) -> str:
        msg = (
            f"Request to change <strong>{self.code.service_func_name}</strong> "
            f"(Pool Id: <strong>{self.code.worker_pool_name}</strong>) "
        )
        msg += "to permission RequestStatus.APPROVED."
        if self.code.nested_codes is None or self.code.nested_codes == {}:  # type: ignore
            msg += " No nested requests"
        else:
            if self.nested_solved:
                # else:
                msg += "<br><br>This change requests the following nested functions calls:<br>"
                msg += self.nested_repr()
            else:
                msg += " Nested Requests not resolved"
        return msg

    def _repr_markdown_(self, wrap_as_python: bool = True, indent: int = 0) -> str:
        code = self.code
        input_policy_type = (
            code.input_policy_type.__canonical_name__
            if code.input_policy_type is not None
            else None
        )
        output_policy_type = (
            code.output_policy_type.__canonical_name__
            if code.output_policy_type is not None
            else None
        )
        repr_dict = {
            "function": code.service_func_name,
            "input_policy_type": f"{input_policy_type}",
            "output_policy_type": f"{output_policy_type}",
            "approved": f"{code.status.approved}",
        }
        return markdown_as_class_with_fields(self, repr_dict)

    @property
    def approved(self) -> bool:
        return self.linked_obj.resolve.approved

    @property
    def valid(self) -> SyftSuccess | SyftError:
        if self.match_type and not isinstance(self.value, UserCodeStatus):
            # TODO: fix the mypy issue
            return SyftError(  # type: ignore[unreachable]
                message=f"{type(self.value)} must be of type: {UserCodeStatus}"
            )
        return SyftSuccess(message=f"{type(self)} valid")

    def mutate(
        self,
        status: UserCodeStatusCollection,
        context: ChangeContext,
        undo: bool,
    ) -> UserCodeStatusCollection | SyftError:
        reason: str = context.extra_kwargs.get("reason", "")

        if not undo:
            res = status.mutate(
                value=(self.value, reason),
                server_name=context.server.name,
                server_id=context.server.id,
                verify_key=context.server.signing_key.verify_key,
            )
            if isinstance(res, SyftError):
                return res
        else:
            res = status.mutate(
                value=(UserCodeStatus.DENIED, reason),
                server_name=context.server.name,
                server_id=context.server.id,
                verify_key=context.server.signing_key.verify_key,
            )
        return res

    def _run(
        self, context: ChangeContext, apply: bool
    ) -> Result[SyftSuccess, SyftError]:
        try:
            valid = self.valid
            if not valid:
                return Err(valid)
            user_code = self.linked_user_code.resolve_with_context(context)
            if user_code.is_err():
                return Err(SyftError(message=user_code.err()))
            user_code = user_code.ok()
            user_code_status = self.linked_obj.resolve_with_context(context)
            if user_code_status.is_err():
                return Err(SyftError(message=user_code_status.err()))
            user_code_status = user_code_status.ok()

            if apply:
                # Only mutate, does not write to stash
                updated_status = self.mutate(user_code_status, context, undo=False)

                if isinstance(updated_status, SyftError):
                    return Err(updated_status.message)

                self.linked_obj.update_with_context(context, updated_status)
            else:
                updated_status = self.mutate(user_code_status, context, undo=True)
                if isinstance(updated_status, SyftError):
                    return Err(updated_status.message)

                self.linked_obj.update_with_context(context, updated_status)
            return Ok(SyftSuccess(message=f"{type(self)} Success"))
        except Exception as e:
            logger.error(f"failed to apply {type(self)}", exc_info=e)
            return Err(SyftError(message=str(e)))

    def apply(self, context: ChangeContext) -> Result[SyftSuccess, SyftError]:
        return self._run(context=context, apply=True)

    def undo(self, context: ChangeContext) -> Result[SyftSuccess, SyftError]:
        return self._run(context=context, apply=False)

    @property
    def link(self) -> SyftObject | None:
        if self.linked_obj:
            return self.linked_obj.resolve
        return None


@serializable()
class SyncedUserCodeStatusChange(UserCodeStatusChange):
    __canonical_name__ = "SyncedUserCodeStatusChange"
    __version__ = SYFT_OBJECT_VERSION_1
    linked_obj: LinkedObject | None = None  # type: ignore

    @property
    def approved(self) -> bool:
        return self.code.status.approved

    def mutate(
        self,
        status: UserCodeStatusCollection,
        context: ChangeContext,
        undo: bool,
    ) -> UserCodeStatusCollection | SyftError:
        return SyftError(
            message="Synced UserCodes status is computed, and cannot be updated manually."
        )

    def _run(
        self, context: ChangeContext, apply: bool
    ) -> Result[SyftSuccess, SyftError]:
        return Ok(
            SyftError(
                message="Synced UserCodes status is computed, and cannot be updated manually."
            )
        )

    def link(self) -> Any:  # type: ignore
        return self.code.status
