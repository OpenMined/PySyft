# stdlib
from collections.abc import Callable
from enum import Enum
import hashlib
import inspect
from typing import Any

# third party
from result import Err
from result import Ok
from result import Result
from typing_extensions import Self

# relative
from ...abstract_node import NodeSideType
from ...client.api import APIRegistry
from ...client.client import SyftClient
from ...custom_worker.config import WorkerConfig
from ...custom_worker.k8s import IN_KUBERNETES
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...serde.serialize import _serialize
from ...store.linked_obj import LinkedObject
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SYFT_OBJECT_VERSION_3
from ...types.syft_object import SyftObject
from ...types.syncable_object import SyncableSyftObject
from ...types.transforms import TransformContext
from ...types.transforms import add_node_uid_for_key
from ...types.transforms import generate_id
from ...types.transforms import transform
from ...types.twin_object import TwinObject
from ...types.uid import LineageID
from ...types.uid import UID
from ...util import options
from ...util.colors import SURFACE
from ...util.markdown import markdown_as_class_with_fields
from ...util.notebook_ui.notebook_addons import REQUEST_ICON
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
from ..job.job_stash import JobInfo
from ..job.job_stash import JobStatus
from ..notification.notifications import Notification
from ..policy.policy import UserPolicy
from ..response import SyftError
from ..response import SyftSuccess
from ..user.user import UserView


@serializable()
class RequestStatus(Enum):
    PENDING = 0
    REJECTED = 1
    APPROVED = 2


@serializable()
class Change(SyftObject):
    __canonical_name__ = "Change"
    __version__ = SYFT_OBJECT_VERSION_2

    linked_obj: LinkedObject | None = None

    def change_object_is_type(self, type_: type) -> bool:
        return self.linked_obj is not None and type_ == self.linked_obj.object_type


@serializable()
class ChangeStatus(SyftObject):
    __canonical_name__ = "ChangeStatus"
    __version__ = SYFT_OBJECT_VERSION_2

    id: UID | None = None  # type: ignore[assignment]
    change_id: UID
    applied: bool = False

    @classmethod
    def from_change(cls, change: Change, applied: bool) -> Self:
        return cls(change_id=change.id, applied=applied)


@serializable()
class ActionStoreChange(Change):
    __canonical_name__ = "ActionStoreChange"
    __version__ = SYFT_OBJECT_VERSION_2

    linked_obj: LinkedObject
    apply_permission_type: ActionPermission

    __repr_attrs__ = ["linked_obj", "apply_permission_type"]

    def _run(
        self, context: ChangeContext, apply: bool
    ) -> Result[SyftSuccess, SyftError]:
        try:
            action_service: ActionService = context.node.get_service(ActionService)  # type: ignore[assignment]
            blob_storage_service = context.node.get_service(BlobStorageService)
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
                requesting_permission_blob_obj = ActionObjectPermission(
                    uid=uid_blob,
                    credentials=context.requesting_user_credentials,
                    permission=self.apply_permission_type,
                )
                if apply:
                    print(
                        "ADDING PERMISSION", requesting_permission_action_obj, id_action
                    )
                    action_store.add_permission(requesting_permission_action_obj)
                    blob_storage_service.stash.add_permission(
                        requesting_permission_blob_obj
                    )
                else:
                    if action_store.has_permission(requesting_permission_action_obj):
                        action_store.remove_permission(requesting_permission_action_obj)
                    if blob_storage_service.stash.has_permission(
                        requesting_permission_blob_obj
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
            print(f"failed to apply {type(self)}", e)
            return Err(SyftError(message=str(e)))

    def apply(self, context: ChangeContext) -> Result[SyftSuccess, SyftError]:
        return self._run(context=context, apply=True)

    def undo(self, context: ChangeContext) -> Result[SyftSuccess, SyftError]:
        return self._run(context=context, apply=False)

    def __repr_syft_nested__(self) -> str:
        return f"Apply <b>{self.apply_permission_type}</b> to \
            <i>{self.linked_obj.object_type.__canonical_name__}:{self.linked_obj.object_uid.short()}</i>"


@serializable()
class CreateCustomImageChange(Change):
    __canonical_name__ = "CreateCustomImageChange"
    __version__ = SYFT_OBJECT_VERSION_2

    config: WorkerConfig
    tag: str
    registry_uid: UID | None = None
    pull_image: bool = True

    __repr_attrs__ = ["config", "tag"]

    def _run(
        self, context: ChangeContext, apply: bool
    ) -> Result[SyftSuccess, SyftError]:
        try:
            worker_image_service = context.node.get_service("SyftWorkerImageService")

            service_context = context.to_service_ctx()
            result = worker_image_service.submit_dockerfile(
                service_context, docker_config=self.config
            )

            if isinstance(result, SyftError):
                return Err(result)

            result = worker_image_service.stash.get_by_docker_config(
                service_context.credentials, config=self.config
            )

            if result.is_err():
                return Err(SyftError(message=f"{result.err()}"))

            worker_image = result.ok()

            build_result = worker_image_service.build(
                service_context,
                image_uid=worker_image.id,
                tag=self.tag,
                registry_uid=self.registry_uid,
                pull=self.pull_image,
            )

            if isinstance(build_result, SyftError):
                return Err(build_result)

            if IN_KUBERNETES:
                push_result = worker_image_service.push(
                    service_context,
                    image=worker_image.id,
                    username=context.extra_kwargs.get("reg_username", None),
                    password=context.extra_kwargs.get("reg_password", None),
                )

                if isinstance(push_result, SyftError):
                    return Err(push_result)

                return Ok(
                    SyftSuccess(
                        message=f"Build Result: {build_result.message} \n Push Result: {push_result.message}"
                    )
                )

            return Ok(build_result)

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
    __version__ = SYFT_OBJECT_VERSION_2

    pool_name: str
    num_workers: int
    image_uid: UID | None = None
    config: WorkerConfig | None = None

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
            worker_pool_service = context.node.get_service("SyftWorkerPoolService")
            service_context: AuthedServiceContext = context.to_service_ctx()

            if self.config is not None:
                result = worker_pool_service.image_stash.get_by_docker_config(
                    service_context.credentials, self.config
                )
                if result.is_err():
                    return Err(SyftError(message=f"{result.err()}"))
                worker_image = result.ok()
                self.image_uid = worker_image.id

            result = worker_pool_service.launch(
                context=service_context,
                name=self.pool_name,
                image_uid=self.image_uid,
                num_workers=self.num_workers,
                reg_username=context.extra_kwargs.get("reg_username", None),
                reg_password=context.extra_kwargs.get("reg_password", None),
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
    __version__ = SYFT_OBJECT_VERSION_2

    requesting_user_verify_key: SyftVerifyKey
    requesting_user_name: str = ""
    requesting_user_email: str | None = ""
    requesting_user_institution: str | None = ""
    approving_user_verify_key: SyftVerifyKey | None = None
    request_time: DateTime
    updated_at: DateTime | None = None
    node_uid: UID
    request_hash: str
    changes: list[Change]
    history: list[ChangeStatus] = []

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
    __exclude_sync_diff_attrs__ = ["node_uid"]

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
            self.node_uid,
            self.syft_client_verify_key,
        )
        shared_with_line = ""
        if self.code and len(self.code.output_readers) > 0:
            # owner_names = ["canada", "US"]
            owners_string = " and ".join(
                [f"<strong>{x}</strong>" for x in self.code.output_reader_names]
            )
            shared_with_line += (
                f"<p><strong>Custom Policy: </strong> "
                f"outputs are <strong>shared</strong> with the owners of {owners_string} once computed"
            )

        if api is not None:
            metadata = api.services.metadata.get_metadata()
            node_name = api.node_name.capitalize() if api.node_name is not None else ""
            node_type = metadata.node_type.value.capitalize()

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
                <p><strong>Requested on: </strong> {node_name} of type <strong> \
                    {node_type}</strong></p>
                <p><strong>Requested by:</strong> {self.requesting_user_name} {email_str} {institution_str}</p>
                <p><strong>Changes: </strong> {str_changes}</p>
            </div>

            """

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
            "Description": " ".join([x.__repr_syft_nested__() for x in self.changes]),
            "Requested By": "\n".join(user_data),
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

    @property
    def code(self) -> Any:
        for change in self.changes:
            if isinstance(change, UserCodeStatusChange):
                return change.code
        return SyftError(
            message="This type of request does not have code associated with it."
        )

    def get_results(self) -> Any:
        return self.code.get_results()

    @property
    def current_change_state(self) -> dict[UID, bool]:
        change_applied_map = {}
        for change_status in self.history:
            # only store the last change
            change_applied_map[change_status.change_id] = change_status.applied

        return change_applied_map

    @property
    def icon(self) -> str:
        return REQUEST_ICON

    @property
    def status(self) -> RequestStatus:
        if len(self.history) == 0:
            return RequestStatus.PENDING

        all_changes_applied = all(self.current_change_state.values()) and (
            len(self.current_change_state) == len(self.changes)
        )

        request_status = (
            RequestStatus.APPROVED if all_changes_applied else RequestStatus.REJECTED
        )

        return request_status

    def approve(
        self,
        disable_warnings: bool = False,
        approve_nested: bool = False,
        **kwargs: dict,
    ) -> Result[SyftSuccess, SyftError]:
        api = APIRegistry.api_for(
            self.node_uid,
            self.syft_client_verify_key,
        )
        if api is None:
            return SyftError(message=f"api is None. You must login to {self.node_uid}")
        # TODO: Refactor so that object can also be passed to generate warnings
        if api.connection:
            metadata = api.connection.get_node_metadata(api.signing_key)
        else:
            metadata = None
        message, is_enclave = None, False

        is_code_request = not isinstance(self.codes, SyftError)

        if is_code_request and len(self.codes) > 1 and not approve_nested:
            return SyftError(
                message="Multiple codes detected, please use approve_nested=True"
            )

        if self.code and not isinstance(self.code, SyftError):
            is_enclave = getattr(self.code, "enclave_metadata", None) is not None

        if is_enclave:
            message = "On approval, the result will be released to the enclave."
        elif metadata and metadata.node_side_type == NodeSideType.HIGH_SIDE.value:
            message = (
                "You're approving a request on "
                f"{metadata.node_side_type} side {metadata.node_type} "
                "which may host datasets with private information."
            )
        if message and metadata and metadata.show_warnings and not disable_warnings:
            prompt_warning_message(message=message, confirm=True)

        print(f"Approving request for domain {api.node_name}")
        res = api.services.request.apply(self.id, **kwargs)
        # if isinstance(res, SyftSuccess):

        return res

    def deny(self, reason: str) -> SyftSuccess | SyftError:
        """Denies the particular request.

        Args:
            reason (str): Reason for which the request has been denied.
        """
        api = APIRegistry.api_for(
            self.node_uid,
            self.syft_client_verify_key,
        )
        if api is None:
            return SyftError(message=f"api is None. You must login to {self.node_uid}")
        return api.services.request.undo(uid=self.id, reason=reason)

    def approve_with_client(self, client: SyftClient) -> Result[SyftSuccess, SyftError]:
        print(f"Approving request for domain {client.name}")
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

        save_method = context.node.get_service_method(RequestService.save)
        return save_method(context=context, request=self)

    def _get_latest_or_create_job(self) -> Job | SyftError:
        """Get the latest job for this requests user_code, or creates one if no jobs exist"""
        api = APIRegistry.api_for(self.node_uid, self.syft_client_verify_key)
        if api is None:
            return SyftError(message=f"api is None. You must login to {self.node_uid}")
        job_service = api.services.job

        existing_jobs = job_service.get_by_user_code_id(self.code.id)
        if isinstance(existing_jobs, SyftError):
            return existing_jobs

        if len(existing_jobs) == 0:
            print("Creating job for existing user code")
            job = job_service.create_job_for_user_code_id(self.code.id)
        else:
            print("returning existing job")
            print("setting permission")
            job = existing_jobs[-1]
            res = job_service.add_read_permission_job_for_code_owner(job, self.code)
            print(res)
            res = job_service.add_read_permission_log_for_code_owner(
                job.log_id, self.code
            )
            print(res)

        return job

    def _is_action_object_from_job(self, action_object: ActionObject) -> Job | None:  # type: ignore
        api = APIRegistry.api_for(self.node_uid, self.syft_client_verify_key)
        if api is None:
            raise ValueError(f"Can't access the api. You must login to {self.node_uid}")
        job_service = api.services.job
        existing_jobs = job_service.get_by_user_code_id(self.code.id)
        for job in existing_jobs:
            if job.result and job.result.id == action_object.id:
                return job

    def accept_by_depositing_result(
        self, result: Any, force: bool = False
    ) -> SyftError | SyftSuccess:
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
            if self._is_action_object_from_job(result):
                action_object_job = self._is_action_object_from_job(result)
                if action_object_job is not None:
                    return SyftError(
                        message=f"This ActionObject is the result of Job {action_object_job.id}, "
                        f"please use the `Job.info` instead."
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

        api = APIRegistry.api_for(self.node_uid, self.syft_client_verify_key)
        if not api:
            raise Exception(
                f"No access to Syft API. Please login to {self.node_uid} first."
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
                    syft_node_location=api.node_uid,
                )
            else:
                action_object = result
            action_object_is_from_this_node = (
                self.syft_node_location == action_object.syft_node_location
            )
            if (
                action_object.syft_blob_storage_entry_id is None
                or not action_object_is_from_this_node
            ):
                action_object.reload_cache()
                action_object.syft_node_location = self.syft_node_location
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
                    syft_node_location=api.node_uid,
                )
            else:
                action_object = result

            # TODO: proper check for if actionobject is already uploaded
            # we also need this for manualy syncing
            action_object_is_from_this_node = (
                self.syft_node_location == action_object.syft_node_location
            )
            if (
                action_object.syft_blob_storage_entry_id is None
                or not action_object_is_from_this_node
            ):
                action_object.reload_cache()
                action_object.syft_node_location = self.syft_node_location
                action_object.syft_client_verify_key = self.syft_client_verify_key
                blob_store_result = action_object._save_to_blob_storage()
                if isinstance(blob_store_result, SyftError):
                    return blob_store_result
                result = api.services.action.set(action_object)
                if isinstance(result, SyftError):
                    return result

            # Do we still need this?
            # policy_state_mutation = ObjectMutation(
            #     linked_obj=user_code_status_change.linked_obj,
            #     attr_name="output_policy",
            #     match_type=True,
            #     value=output_policy,
            # )

            action_object_link = LinkedObject.from_obj(result, node_uid=self.node_uid)
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

            res = api.services.code.store_as_history(
                user_code_id=code.id,
                outputs=result,
                job_id=job.id,
                input_ids=input_ids,
            )
            if isinstance(res, SyftError):
                return res

        job_info.result = action_object

        existing_result = job.result.id if job.result is not None else None
        print(
            f"Job({job.id}) Setting new result {existing_result} -> {job_info.result.id}"
        )
        job.apply_info(job_info)

        job_service = api.services.job
        res = job_service.update(job)
        if isinstance(res, SyftError):
            return res

        return SyftSuccess(message="Request submitted for updating result.")

    def sync_job(
        self, job_info: JobInfo, **kwargs: Any
    ) -> Result[SyftSuccess, SyftError]:
        if job_info.includes_result:
            return SyftError(
                message="This JobInfo includes a Result. Please use Request.accept_by_depositing_result instead."
            )

        api = APIRegistry.api_for(
            node_uid=self.node_uid, user_verify_key=self.syft_client_verify_key
        )
        if api is None:
            return SyftError(message=f"api is None. You must login to {self.node_uid}")
        job_service = api.services.job

        job = self._get_latest_or_create_job()
        job.apply_info(job_info)
        return job_service.update(job)

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
    __version__ = SYFT_OBJECT_VERSION_2

    user: UserView
    request: Request
    message: Notification


@serializable()
class RequestInfoFilter(SyftObject):
    # version
    __canonical_name__ = "RequestInfoFilter"
    __version__ = SYFT_OBJECT_VERSION_2

    name: str | None = None


@serializable()
class SubmitRequest(SyftObject):
    __canonical_name__ = "SubmitRequest"
    __version__ = SYFT_OBJECT_VERSION_2

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
    if context.output and context.node and context.obj:
        if context.obj.requesting_user_verify_key and context.node.is_root(
            context.credentials
        ):
            context.output["requesting_user_verify_key"] = (
                context.obj.requesting_user_verify_key
            )
        else:
            context.output["requesting_user_verify_key"] = context.credentials

    return context


def add_requesting_user_info(context: TransformContext) -> TransformContext:
    if context.output is not None and context.node is not None:
        try:
            user_key = context.output["requesting_user_verify_key"]
            user_service = context.node.get_service("UserService")
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
        add_node_uid_for_key("node_uid"),
        add_request_time,
        check_requesting_user_verify_key,
        add_requesting_user_info,
        hash_changes,
    ]


@serializable()
class ObjectMutation(Change):
    __canonical_name__ = "ObjectMutation"
    __version__ = SYFT_OBJECT_VERSION_2

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
    __version__ = SYFT_OBJECT_VERSION_2

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
    __version__ = SYFT_OBJECT_VERSION_3

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
        return self.linked_user_code.resolve

    @property
    def codes(self) -> list[UserCode]:
        def recursive_code(node: Any) -> list:
            codes = []
            for _, (obj, new_node) in node.items():
                codes.append(obj.resolve)
                codes.extend(recursive_code(new_node))
            return codes

        codes = [self.code]
        codes.extend(recursive_code(self.code.nested_codes))
        return codes

    def nested_repr(self, node: Any | None = None, level: int = 0) -> str:
        msg = ""
        if node is None:
            node = self.code.nested_codes

        for service_func_name, (_, new_node) in node.items():  # type: ignore
            msg = "├──" + "──" * level + f"{service_func_name}<br>"
            msg += self.nested_repr(node=new_node, level=level + 1)
        return msg

    def __repr_syft_nested__(self) -> str:
        msg = (
            f"Request to change <b>{self.code.service_func_name}</b> "
            f"(Pool Id: <b>{self.code.worker_pool_name}</b>) "
        )
        msg += "to permission <b>RequestStatus.APPROVED</b>"
        if self.nested_solved:
            if self.link.nested_codes == {}:  # type: ignore
                msg += ". No nested requests"
            else:
                msg += ".<br><br>This change requests the following nested functions calls:<br>"
                msg += self.nested_repr()
        else:
            msg += ". Nested Requests not resolved"
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

    # def get_nested_requests(self, context, code_tree: Dict[str: Tuple[LinkedObject, Dict]]):
    #     approved_nested_codes = {}
    #     for key, (linked_obj, new_code_tree) in code_tree.items():
    #         code_obj = linked_obj.resolve_with_context(context).ok()
    #         approved_nested_codes[key] = code_obj.id

    #         res = self.get_nested_requests(context, new_code_tree)
    #         if isinstance(res, SyftError):
    #             return res
    #         code_obj.nested_codes = res
    #         linked_obj.update_with_context(context, code_obj)

    #     return approved_nested_codes

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
                node_name=context.node.name,
                node_id=context.node.id,
                verify_key=context.node.signing_key.verify_key,
            )
            if isinstance(res, SyftError):
                return res
        else:
            res = status.mutate(
                value=(UserCodeStatus.DENIED, reason),
                node_name=context.node.name,
                node_id=context.node.id,
                verify_key=context.node.signing_key.verify_key,
            )
        return res

    def is_enclave_request(self, user_code: UserCode) -> bool:
        return (
            user_code.is_enclave_code is not None
            and self.value == UserCodeStatus.APPROVED
        )

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

                # relative
                from ..enclave.enclave_service import propagate_inputs_to_enclave

                self.linked_obj.update_with_context(context, updated_status)
                if self.is_enclave_request(user_code):
                    enclave_res = propagate_inputs_to_enclave(
                        user_code=user_code, context=context
                    )
                    if isinstance(enclave_res, SyftError):
                        return enclave_res
            else:
                updated_status = self.mutate(user_code_status, context, undo=True)
                if isinstance(updated_status, SyftError):
                    return Err(updated_status.message)

                # TODO: Handle Enclave approval.
                self.linked_obj.update_with_context(context, updated_status)
            return Ok(SyftSuccess(message=f"{type(self)} Success"))
        except Exception as e:
            print(f"failed to apply {type(self)}. {e}")
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
