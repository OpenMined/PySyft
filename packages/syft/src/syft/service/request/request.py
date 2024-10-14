# stdlib
from collections.abc import Callable
from enum import Enum
import hashlib
import inspect
import logging
from typing import Any

# third party
from pydantic import model_validator
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
from ...types.errors import SyftException
from ...types.result import Err
from ...types.result import as_result
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
from ...util.decorators import deprecated
from ...util.markdown import markdown_as_class_with_fields
from ...util.notebook_ui.icons import Icon
from ...util.util import prompt_warning_message
from ..action.action_object import ActionObject
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from ..code.user_code import ApprovalDecision
from ..code.user_code import UserCode
from ..code.user_code import UserCodeStatus
from ..code.user_code import UserCodeStatusCollection
from ..context import AuthedServiceContext
from ..context import ChangeContext
from ..job.job_stash import Job
from ..job.job_stash import JobInfo
from ..job.job_stash import JobStatus
from ..notification.notifications import Notification
from ..policy.policy import UserPolicy
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
    def from_usercode_status(
        cls, status: UserCodeStatusCollection, context: AuthedServiceContext
    ) -> "RequestStatus":
        if status.get_is_approved(context):
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

    @as_result(SyftException)
    def _run(self, context: ChangeContext, apply: bool) -> SyftSuccess:
        action_store = context.server.services.action.stash

        # can we ever have a lineage ID in the store?
        obj_uid = self.linked_obj.object_uid
        obj_uid = obj_uid.id if isinstance(obj_uid, LineageID) else obj_uid

        action_obj = action_store.get(
            uid=obj_uid,
            credentials=context.approving_user_credentials,
        ).unwrap()

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
                uid_blob = action_obj.syft_blob_storage_entry_id  # type: ignore[unreachable]

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
                    context.server.services.blob_storage.stash.add_permission(
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
                    and context.server.services.blob_storage.stash.has_permission(
                        requesting_permission_blob_obj
                    )
                ):
                    context.server.services.blob_storage.stash.remove_permission(
                        requesting_permission_blob_obj
                    )
        else:
            raise SyftException(
                public_message=f"No permission for approving_user_credentials {context.approving_user_credentials}"
            )

        return SyftSuccess(message=f"{type(self)} Success")

    @as_result(SyftException)
    def apply(self, context: ChangeContext) -> SyftSuccess:
        return self._run(context=context, apply=True).unwrap()

    @as_result(SyftException)
    def undo(self, context: ChangeContext) -> SyftSuccess:
        return self._run(context=context, apply=False).unwrap()

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

    @as_result(SyftException)
    def _run(self, context: ChangeContext, apply: bool) -> SyftSuccess:
        service_context = context.to_service_ctx()
        context.server.services.syft_worker_image.submit(
            service_context, worker_config=self.config
        )

        worker_image = (
            context.server.services.syft_worker_image.stash.get_by_worker_config(
                service_context.credentials, config=self.config
            ).unwrap()
        )
        if worker_image is None:
            raise SyftException(public_message="The worker image does not exist.")

        build_success_message = "Image was pre-built."

        if not worker_image.is_prebuilt:
            build_result = context.server.services.syft_worker_image.build(
                service_context,
                image_uid=worker_image.id,
                tag=self.tag,
                registry_uid=self.registry_uid,
                pull_image=self.pull_image,
            )
            build_success_message = build_result.message

        build_success = f"Build result: {build_success_message}"
        if IN_KUBERNETES and not worker_image.is_prebuilt:
            push_result = context.server.services.syft_worker_image.push(
                service_context,
                image_uid=worker_image.id,
                username=context.extra_kwargs.get("registry_username", None),
                password=context.extra_kwargs.get("registry_password", None),
            )
            return SyftSuccess(
                message=f"{build_success}\nPush result: {push_result.message}"
            )

        return SyftSuccess(message=build_success)

    @as_result(SyftException)
    def apply(self, context: ChangeContext) -> SyftSuccess:
        return self._run(context=context, apply=True).unwrap()

    @as_result(SyftException)
    def undo(self, context: ChangeContext) -> SyftSuccess:
        return self._run(context=context, apply=False).unwrap()

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

    @as_result(SyftException)
    def _run(self, context: ChangeContext, apply: bool) -> SyftSuccess:
        """
        This function is run when the DO approves (apply=True)
        or deny (apply=False) the request.
        """
        if apply:
            # get the worker pool service and try to launch a pool
            service_context: AuthedServiceContext = context.to_service_ctx()

            if self.config is not None:
                worker_image = context.server.services.syft_worker_pool.image_stash.get_by_worker_config(
                    service_context.credentials, self.config
                ).unwrap()
                self.image_uid = worker_image.id

            result = context.server.services.syft_worker_pool.launch(
                context=service_context,
                pool_name=self.pool_name,
                image_uid=self.image_uid,
                num_workers=self.num_workers,
                registry_username=context.extra_kwargs.get("registry_username", None),
                registry_password=context.extra_kwargs.get("registry_password", None),
                pod_annotations=self.pod_annotations,
                pod_labels=self.pod_labels,
            )
            return SyftSuccess(message="Worker successfully launched", value=result)
        else:
            raise SyftException(
                public_message=f"Request to create a worker pool with name {self.name} denied"
            )

    @as_result(SyftException)
    def apply(self, context: ChangeContext) -> SyftSuccess:
        return self._run(context=context, apply=True).unwrap()

    @as_result(SyftException)
    def undo(self, context: ChangeContext) -> SyftSuccess:
        return self._run(context=context, apply=False).unwrap()

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
        "code_id",
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

        api = self.get_api_wrapped()
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
        if api.is_ok():
            api = api.unwrap()
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

    @property
    def deny_reason(self) -> str | SyftError:
        code = self.code
        if isinstance(code, SyftError):
            return code

        code_status: UserCodeStatusCollection = code.status_link.resolve
        return code_status.first_denial_reason

    @as_result(SyftException)
    def get_deny_reason(self, context: AuthedServiceContext) -> str | None:
        code = self.get_user_code(context).unwrap()
        if code is None:
            return None

        code_status = code.get_status(context).unwrap()
        return code_status.first_denial_reason

    def _coll_repr_(self) -> dict[str, str | dict[str, str]]:
        # relative
        from ...util.notebook_ui.components.sync import Badge

        if self.status == RequestStatus.APPROVED:
            badge_color = "badge-green"
        elif self.status == RequestStatus.PENDING:
            badge_color = "badge-gray"
        else:
            badge_color = "badge-red"

        status_badge = Badge(
            value=self.status.name.capitalize(),
            badge_class=badge_color,
        ).to_html()

        if self.status == RequestStatus.REJECTED:
            deny_reason = self.deny_reason
            if isinstance(deny_reason, str) and len(deny_reason) > 0:
                status_badge += (
                    "<br><span style='margin-top: 8px; display: block;'>"
                    f"<strong>Deny Reason:</strong> {deny_reason}</span>"
                )

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
        raise SyftException(
            public_message="This type of request does not have code associated with it."
        )

    @property
    def status_id(self) -> UID:
        for change in self.changes:
            if isinstance(change, UserCodeStatusChange):
                return change.linked_obj.object_uid  # type: ignore
        raise SyftException(
            public_message="This type of request does not have code associated with it."
        )

    @property
    def codes(self) -> Any:
        for change in self.changes:
            if isinstance(change, UserCodeStatusChange):
                return change.codes
        return SyftError(
            message="This type of request does not have code associated with it."
        )

    @as_result(SyftException)
    def get_user_code(self, context: AuthedServiceContext) -> UserCode | None:
        for change in self.changes:
            if isinstance(change, UserCodeStatusChange):
                return change.get_user_code(context).unwrap()
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
        # TODO fix
        try:
            # this is breaking in l2 coming from sending a request email to admin
            is_l0_deployment = (
                self.get_is_l0_deployment(context) if context else self.is_l0_deployment
            )
            if is_l0_deployment:
                code_status = (
                    self.code.get_status(context).unwrap()
                    if context
                    else self.code.status
                )
                return RequestStatus.from_usercode_status(code_status, context)
        except Exception:  # nosec
            # this breaks when coming from a user submitting a request
            # which tries to send an email to the admin and ends up here
            pass  # lets keep going

        self.refresh()
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
    ) -> SyftSuccess:
        api = self._get_api()

        if self.is_l0_deployment:
            raise SyftException(
                public_message="This request is a low-side request. Please sync your results to approve."
            )
        # TODO: Refactor so that object can also be passed to generate warnings
        if api.connection:
            metadata = api.connection.get_server_metadata(api.signing_key)
        else:
            metadata = None
        message = None

        is_code_request = not isinstance(self.codes, SyftError)

        if is_code_request and len(self.codes) > 1 and not approve_nested:
            raise SyftException(
                public_message="Multiple codes detected, please use approve_nested=True"
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

    def deny(self, reason: str) -> SyftSuccess:
        """Denies the particular request.

        Args:
            reason (str): Reason for which the request has been denied.
        """
        api = self._get_api()

        if self.is_l0_deployment:
            if self.status == RequestStatus.APPROVED:
                prompt_warning_message(
                    "This request already has results published to the data scientist. "
                    "They will still be able to access those results."
                )
            api.code_status.update(
                id=self.code.status_link.object_uid,
                decision=ApprovalDecision(status=UserCodeStatus.DENIED, reason=reason),
            )

            return SyftSuccess(message=f"Request denied with reason: {reason}")

        return api.services.request.undo(uid=self.id, reason=reason)

    @property
    def is_l0_deployment(self) -> bool:
        return bool(self.code) and self.code.is_l0_deployment

    def get_is_l0_deployment(self, context: AuthedServiceContext) -> bool:
        code = self.get_user_code(context).unwrap()
        if code:
            return code.is_l0_deployment
        else:
            return False

    def approve_with_client(self, client: SyftClient) -> SyftSuccess:
        if self.is_l0_deployment:
            raise SyftException(
                public_message="This request is a low-side request. Please sync your results to approve."
            )

        print(f"Approving request for datasite {client.name}")
        return client.api.services.request.apply(self.id)

    @as_result(SyftException)
    def apply(self, context: AuthedServiceContext) -> SyftSuccess:
        change_context: ChangeContext = ChangeContext.from_service(context)
        change_context.requesting_user_credentials = self.requesting_user_verify_key

        for change in self.changes:
            # by default change status is not applied
            change_status = ChangeStatus(change_id=change.id, applied=False)

            # FIX: Change change.apply
            try:
                change.apply(context=change_context).unwrap()
                change_status.applied = True
                self.history.append(change_status)
            except:
                # add to history and save history to request
                self.history.append(change_status)
                self.save(context=context)
                raise

        self.updated_at = DateTime.now()
        self.save(context=context)

        return SyftSuccess(message=f"Request {self.id} changes applied")

    def undo(self, context: AuthedServiceContext) -> SyftSuccess:
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

        # override object with latest changes.
        self = result
        return SyftSuccess(message=f"Request {self.id} changes undone.", value=self.id)

    def save(self, context: AuthedServiceContext) -> SyftSuccess:
        # relative

        return context.server.services.request.save(context=context, request=self)

    def _create_action_object_for_deposited_result(
        self,
        result: Any,
    ) -> ActionObject:
        api = self._get_api()

        # Ensure result is an ActionObject
        if isinstance(result, ActionObject):
            try:
                existing_job = api.services.job.get_by_result_id(result.id.id)
            except SyftException:
                existing_job = None
            if existing_job is not None:
                raise SyftException(
                    public_message=f"This ActionObject is already the result of Job {existing_job.id}"
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
            action_object._send(self.server_uid, self.syft_client_verify_key)
        return action_object

    def _create_output_history_for_deposited_result(
        self, job: Job, result: Any
    ) -> SyftSuccess:
        code = self.code
        api = self._get_api()
        input_ids = {}
        input_policy = code.input_policy
        if input_policy is not None:
            for input_ in input_policy.inputs.values():
                input_ids.update(input_)

        input_ids = {k: v for k, v in input_ids.items() if isinstance(v, UID)}

        return api.services.code.store_execution_output(
            user_code_id=code.id,
            outputs=result,
            job_id=job.id,
            input_ids=input_ids,
        )

    def deposit_result(
        self,
        result: Any,
        log_stdout: str = "",
        log_stderr: str = "",
        approve: bool | None = None,
        **kwargs: dict[str, Any],
    ) -> Job:
        """
        Adds a result to this Request:
        - Create an ActionObject from the result (if not already an ActionObject)
        - Ensure ActionObject exists on this server
        - Create Job with new result and logs
        - Update the output history

        If this is a L2 request, the old accept_by_deposit_result will be used.

        Args:
            result (Any): ActionObject or any object to be saved as an ActionObject.
            log_stdout (str): stdout logs.
            log_stderr (str): stderr logs.
            approve (bool, optional): Only supported for L2 requests. If True, the request will be approved.
                Defaults to None.


        Returns:
            Job: Job object if successful, else raise SyftException.
        """

        # L2 request
        # TODO specify behavior and rewrite old flow
        if not self.is_l0_deployment:
            if approve is None:
                approve = prompt_warning_message(
                    "Depositing a result on this request will approve it.",
                    confirm=True,
                )
            if approve is False:
                raise SyftException(
                    public_message="Cannot deposit result without approving the request."
                )
            else:
                return self._deposit_result_l2(result, **kwargs)

        # L0 request
        if approve:
            return SyftError(
                message="This is a request from the low side, it can only be approved by syncing the results."
            )

        api = self._get_api()
        if isinstance(api, SyftError):
            return api
        code = self.code
        if isinstance(code, SyftError):
            return code

        # Create ActionObject
        action_object = self._create_action_object_for_deposited_result(result)
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
        # Add to output history
        self._create_output_history_for_deposited_result(job, action_object)
        return job

    def _get_job_from_action_object(self, action_object: ActionObject) -> Job | None:
        api = self._get_api()
        if isinstance(api, SyftError):
            return None

        job = api.services.job.get_by_result_id(action_object.id.id)
        return job

    def _get_latest_or_create_job(self) -> Job | SyftError:
        """Get the latest job for this requests user_code, or creates one if no jobs exist"""
        api = self._get_api()
        if isinstance(api, SyftError):
            return api
        job_service = api.services.job

        existing_jobs = job_service.get_by_user_code_id(self.code.id)
        if isinstance(existing_jobs, SyftError):
            return existing_jobs

        if len(existing_jobs) == 0:
            job = job_service.create_job_for_user_code_id(
                user_code_id=self.code.id,
                add_code_owner_read_permissions=True,
            )
        else:
            job = existing_jobs[-1]
            res = job_service.add_read_permission_job_for_code_owner(job, self.code)
            res = job_service.add_read_permission_log_for_code_owner(
                job.log_id, self.code
            )
            print(res)

        return job

    def _deposit_result_l2(
        self,
        result: Any,
        force: bool = False,
    ) -> Job | SyftError:
        # this code is extremely brittle because its a work around that relies on
        # the type of request being very specifically tied to code which needs approving

        # Special case for results from Jobs (High-low side async)
        if isinstance(result, JobInfo):
            job_info = result
            if not job_info.includes_result:
                return SyftError(
                    message="JobInfo should not include result. Use sync_job instead."
                )
            result = job_info.result
        elif isinstance(result, ActionObject):
            # Do not allow accepting a result produced by a Job,
            # This can cause an inconsistent Job state
            action_object_job = self._get_job_from_action_object(result)
            if action_object_job is not None:
                return SyftError(
                    message=f"This ActionObject is the result of Job {action_object_job.id}, "
                    f"please use the `Job.info` instead."
                )
            else:
                job_info = JobInfo(
                    includes_metadata=True,
                    includes_result=True,
                    status=JobStatus.COMPLETED,
                    resolved=True,
                )
        else:
            # NOTE result is added at the end of function (once ActionObject is created)
            job_info = JobInfo(
                includes_metadata=True,
                includes_result=True,
                status=JobStatus.COMPLETED,
                resolved=True,
            )

        user_code_status_change: UserCodeStatusChange = self.changes[0]
        code = user_code_status_change.code
        output_history = code.output_history
        if isinstance(output_history, SyftError):
            return output_history
        output_policy = code.output_policy
        if isinstance(output_policy, SyftError):
            return output_policy
        if isinstance(user_code_status_change.code.output_policy_type, UserPolicy):
            return SyftError(
                message="UserCode uses an user-submitted custom policy. Please use .approve()"
            )

        if not user_code_status_change.change_object_is_type(UserCodeStatusCollection):
            raise TypeError(
                f"accept_by_depositing_result can only be run on {UserCodeStatusCollection} not "
                f"{user_code_status_change.linked_obj.object_type}"
            )
        if not type(user_code_status_change) == UserCodeStatusChange:
            raise TypeError(
                f"accept_by_depositing_result can only be run on {UserCodeStatusChange} not "
                f"{type(user_code_status_change)}"
            )

        api = APIRegistry.api_for(self.server_uid, self.syft_client_verify_key).unwrap()
        if not api:
            raise Exception(
                f"No access to Syft API. Please login to {self.server_uid} first."
            )
        if api.signing_key is None:
            raise ValueError(f"{api}'s signing key is None")
        is_approved = user_code_status_change.approved

        permission_request = self.approve(approve_nested=True)
        if isinstance(permission_request, SyftError):
            return permission_request

        job = self._get_latest_or_create_job()
        if isinstance(job, SyftError):
            return job

        # This weird order is due to the fact that state is None before calling approve
        # we could fix it in a future release
        if is_approved:
            if not force:
                return SyftError(
                    message="Already approved, if you want to force updating the result use force=True"
                )
            # TODO: this should overwrite the output history instead
            action_obj_id = output_history[0].output_ids[0]  # type: ignore

            if not isinstance(result, ActionObject):
                action_object = ActionObject.from_obj(
                    result,
                    id=action_obj_id,
                    syft_client_verify_key=api.signing_key.verify_key,
                    syft_server_location=api.server_uid,
                )
            else:
                action_object = result
            action_object_is_from_this_node = (
                self.syft_server_location == action_object.syft_server_location
            )
            if (
                action_object.syft_blob_storage_entry_id is None
                or not action_object_is_from_this_node
            ):
                action_object.reload_cache()
                action_object.syft_server_location = self.syft_server_location
                action_object.syft_client_verify_key = self.syft_client_verify_key
                blob_store_result = action_object._save_to_blob_storage()
                if isinstance(blob_store_result, SyftError):
                    return blob_store_result
                result = api.services.action.set(action_object)
                if isinstance(result, SyftError):
                    return result
        else:
            if not isinstance(result, ActionObject):
                action_object = ActionObject.from_obj(
                    result,
                    syft_client_verify_key=api.signing_key.verify_key,
                    syft_server_location=api.server_uid,
                )
            else:
                action_object = result

            # TODO: proper check for if actionobject is already uploaded
            # we also need this for manualy syncing
            action_object_is_from_this_node = (
                self.syft_server_location == action_object.syft_server_location
            )
            if (
                action_object.syft_blob_storage_entry_id is None
                or not action_object_is_from_this_node
            ):
                action_object.reload_cache()
                action_object.syft_server_location = self.syft_server_location
                action_object.syft_client_verify_key = self.syft_client_verify_key
                blob_store_result = action_object._save_to_blob_storage()
                if isinstance(blob_store_result, SyftError):
                    return blob_store_result
                result = api.services.action.set(action_object)
                if isinstance(result, SyftError):
                    return result

            action_object_link = LinkedObject.from_obj(
                result, server_uid=self.server_uid
            )
            permission_change = ActionStoreChange(
                linked_obj=action_object_link,
                apply_permission_type=ActionPermission.READ,
            )

            new_changes = [permission_change]
            result_request = api.services.request.add_changes(
                uid=self.id, changes=new_changes
            )
            if isinstance(result_request, SyftError):
                return result_request
            self = result_request

            approved = self.approve(disable_warnings=True, approve_nested=True)
            if isinstance(approved, SyftError):
                return approved

            input_ids = {}
            if code.input_policy is not None:
                for inps in code.input_policy.inputs.values():
                    input_ids.update(inps)

            input_ids = {k: v for k, v in input_ids.items() if isinstance(v, UID)}
            res = api.services.code.store_execution_output(
                user_code_id=code.id,
                outputs=result,
                job_id=job.id,
                input_ids=input_ids,
            )
            if isinstance(res, SyftError):
                return res

        job_info.result = action_object
        job_info.status = (
            JobStatus.ERRORED
            if isinstance(action_object.syft_action_data, Err)
            else JobStatus.COMPLETED
        )

        existing_result = None
        if isinstance(job.result, ActionObject):
            existing_result = job.result.id
        elif isinstance(job.result, Err):
            existing_result = job.result  # type: ignore [assignment]
        else:
            existing_result = job.result
        print(
            f"Job({job.id}) Setting new result {existing_result} -> {job_info.result.id}"
        )
        job.apply_info(job_info)

        job_service = api.services.job
        res = job_service.update(job)
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

    def get_sync_dependencies(self, context: AuthedServiceContext) -> list[UID]:
        return [self.code_id, self.status_id]


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
            user = context.server.services.user.get_by_verify_key(user_key).unwrap()
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

    @as_result(SyftException)
    def _run(self, context: ChangeContext, apply: bool) -> SyftSuccess:
        if self.linked_obj is None:
            raise SyftException(public_message=f"{self}'s linked object is None")

        obj = self.linked_obj.resolve_with_context(context).unwrap()

        if apply:
            obj = self.mutate(obj, value=self.value)
            self.linked_obj.update_with_context(context, obj)
        else:
            # unset the set value
            obj = self.mutate(obj, value=self.previous_value)
            self.linked_obj.update_with_context(context, obj)

        return SyftSuccess(message=f"{type(self)} Success")

    @as_result(SyftException)
    def apply(self, context: ChangeContext) -> SyftSuccess:
        return self._run(context=context, apply=True).unwrap()

    @as_result(SyftException)
    def undo(self, context: ChangeContext) -> SyftSuccess:
        return self._run(context=context, apply=False).unwrap()


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
    def valid(self) -> SyftSuccess:
        if self.match_type and not isinstance(self.value, self.enum_type):
            raise SyftException(
                public_message=f"{type(self.value)} must be of type: {self.enum_type}"
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

    @as_result(SyftException)
    def _run(self, context: ChangeContext, apply: bool) -> SyftSuccess:
        valid = self.valid
        if not valid:
            raise SyftException(public_message=valid.message)
        if self.linked_obj is None:
            raise SyftException(public_message=f"{self}'s linked object is None")
        obj = self.linked_obj.resolve_with_context(context).unwrap()

        if apply:
            obj = self.mutate(obj=obj)
            self.linked_obj.update_with_context(context, obj)
        else:
            raise SyftException(public_message="undo not implemented")

        return SyftSuccess(message=f"{type(self)} Success")

    @as_result(SyftException)
    def apply(self, context: ChangeContext) -> SyftSuccess:
        return self._run(context=context, apply=True).unwrap()

    @as_result(SyftException)
    def undo(self, context: ChangeContext) -> SyftSuccess:
        return self._run(context=context, apply=False).unwrap()

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

    @as_result(SyftException)
    def get_user_code(self, context: AuthedServiceContext) -> UserCode:
        return self.linked_user_code.resolve_with_context(context).unwrap()

    @property
    def codes(self) -> list[UserCode]:
        def recursive_code(server: Any) -> list:
            codes = []
            for obj, new_server in server.values():
                # TODO: this fixes problems with getting the api for object
                # we should fix this more properly though
                obj.syft_server_location = obj.server_uid
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
    def valid(self) -> SyftSuccess:
        if self.match_type and not isinstance(self.value, UserCodeStatus):
            # TODO: fix the mypy issue
            raise SyftException(  # type: ignore[unreachable]
                public_message=f"{type(self.value)} must be of type: {UserCodeStatus}"
            )
        return SyftSuccess(message=f"{type(self)} valid")

    @as_result(SyftException)
    def mutate(
        self,
        status: UserCodeStatusCollection,
        context: ChangeContext,
        undo: bool,
    ) -> UserCodeStatusCollection:
        reason: str = context.extra_kwargs.get("reason", "")
        return status.mutate(
            value=ApprovalDecision(
                status=UserCodeStatus.DENIED if undo else self.value, reason=reason
            ),
            server_name=context.server.name,
            server_id=context.server.id,
            verify_key=context.server.signing_key.verify_key,
        ).unwrap()

    @as_result(SyftException)
    def _run(self, context: ChangeContext, apply: bool) -> SyftSuccess:
        valid = self.valid

        if not valid:
            raise SyftException(public_message=valid.message)

        self.linked_user_code.resolve_with_context(context).unwrap()
        user_code_status = self.linked_obj.resolve_with_context(context).unwrap()

        if apply:
            # Only mutate, does not write to stash
            updated_status = self.mutate(user_code_status, context, undo=False).unwrap()
            self.linked_obj.update_with_context(context, updated_status)
        else:
            updated_status = self.mutate(user_code_status, context, undo=True).unwrap()
            self.linked_obj.update_with_context(context, updated_status)
        return SyftSuccess(message=f"{type(self)} Success")

    @as_result(SyftException)
    def apply(self, context: ChangeContext) -> SyftSuccess:
        return self._run(context=context, apply=True).unwrap()

    @as_result(SyftException)
    def undo(self, context: ChangeContext) -> SyftSuccess:
        return self._run(context=context, apply=False).unwrap()

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
    ) -> UserCodeStatusCollection:
        raise SyftException(
            public_message="Synced UserCodes status is computed, and cannot be updated manually."
        )

    @as_result(SyftException)
    def _run(self, context: ChangeContext, apply: bool) -> SyftSuccess:
        raise SyftException(
            public_message="Synced UserCodes status is computed, and cannot be updated manually."
        )

    def link(self) -> Any:  # type: ignore
        return self.code.status
