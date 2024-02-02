# stdlib
from enum import Enum
import hashlib
import inspect
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from result import Err
from result import Ok
from result import Result
from typing_extensions import Self

# relative
from ...abstract_node import NodeSideType
from ...client.api import APIRegistry
from ...custom_worker.config import WorkerConfig
from ...custom_worker.k8s import IN_KUBERNETES
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...serde.serialize import _serialize
from ...store.linked_obj import LinkedObject
from ...types.datetime import DateTime
from ...types.syft_migration import migrate
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject
from ...types.transforms import TransformContext
from ...types.transforms import add_node_uid_for_key
from ...types.transforms import drop
from ...types.transforms import generate_id
from ...types.transforms import make_set_default
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
from ..context import AuthedServiceContext
from ..context import ChangeContext
from ..job.job_stash import Job
from ..job.job_stash import JobInfo
from ..job.job_stash import JobStatus
from ..notification.notifications import Notification
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
    __version__ = SYFT_OBJECT_VERSION_1

    linked_obj: Optional[LinkedObject]

    def is_type(self, type_: type) -> bool:
        return self.linked_obj and type_ == self.linked_obj.object_type


@serializable()
class ChangeStatus(SyftObject):
    __canonical_name__ = "ChangeStatus"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
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
            action_service: ActionService = context.node.get_service(ActionService)
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

    def __repr_syft_nested__(self):
        return f"Apply <b>{self.apply_permission_type}</b> to \
            <i>{self.linked_obj.object_type.__canonical_name__}:{self.linked_obj.object_uid.short()}</i>"


@serializable()
class CreateCustomImageChange(Change):
    __canonical_name__ = "CreateCustomImageChange"
    __version__ = SYFT_OBJECT_VERSION_1

    config: WorkerConfig
    tag: str
    registry_uid: Optional[UID]

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

    def __repr_syft_nested__(self):
        return f"Create Image for Config: {self.config} with tag: {self.tag}"


@serializable()
class CreateCustomWorkerPoolChange(Change):
    __canonical_name__ = "CreateCustomWorkerPoolChange"
    __version__ = SYFT_OBJECT_VERSION_1

    pool_name: str
    num_workers: int
    image_uid: Optional[UID]
    config: Optional[WorkerConfig]

    __repr_attrs__ = ["pool_name", "num_workers", "image_uid"]

    def _run(
        self, context: ChangeContext, apply: bool
    ) -> Result[SyftSuccess, SyftError]:
        """
        This function is run when the DO approves (apply=True)
        or deny (apply=False) the request.
        """
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

    def __repr_syft_nested__(self):
        return (
            f"Create Worker Pool '{self.pool_name}' for Image with id {self.image_uid}"
        )


@serializable()
class Request(SyftObject):
    __canonical_name__ = "Request"
    __version__ = SYFT_OBJECT_VERSION_1

    requesting_user_verify_key: SyftVerifyKey
    requesting_user_name: str = ""
    requesting_user_email: Optional[str] = ""
    requesting_user_institution: Optional[str] = ""
    approving_user_verify_key: Optional[SyftVerifyKey]
    request_time: DateTime
    updated_at: Optional[DateTime]
    node_uid: UID
    request_hash: str
    changes: List[Change]
    history: List[ChangeStatus] = []

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

    def _repr_html_(self) -> Any:
        # add changes
        updated_at_line = ""
        if self.updated_at is not None:
            updated_at_line += (
                f"<p><strong>Created by: </strong>{self.requesting_user_name}</p>"
            )
        str_changes = []
        for change in self.changes:
            str_change = (
                change.__repr_syft_nested__()
                if hasattr(change, "__repr_syft_nested__")
                else type(change)
            )
            str_change = f"{str_change}. "
            str_changes.append(str_change)
        str_changes = "\n".join(str_changes)
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

        metadata = api.services.metadata.get_metadata()
        node_name = api.node_name.capitalize() if api.node_name is not None else ""

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
                    {metadata.node_type.value.capitalize()}</strong></p>
                <p><strong>Requested by:</strong> {self.requesting_user_name} {email_str} {institution_str}</p>
                <p><strong>Changes: </strong> {str_changes}</p>
            </div>

            """

    def _coll_repr_(self):
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
    def current_change_state(self) -> Dict[UID, bool]:
        change_applied_map = {}
        for change_status in self.history:
            # only store the last change
            change_applied_map[change_status.change_id] = change_status.applied

        return change_applied_map

    @property
    def icon(self):
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
    ):
        api = APIRegistry.api_for(
            self.node_uid,
            self.syft_client_verify_key,
        )
        # TODO: Refactor so that object can also be passed to generate warnings
        metadata = api.connection.get_node_metadata(api.signing_key)
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
        elif metadata.node_side_type == NodeSideType.HIGH_SIDE.value:
            message = (
                "You're approving a request on "
                f"{metadata.node_side_type} side {metadata.node_type} "
                "which may host datasets with private information."
            )
        if message and metadata.show_warnings and not disable_warnings:
            prompt_warning_message(message=message, confirm=True)

        print(f"Approving request for domain {api.node_name}")
        return api.services.request.apply(self.id, **kwargs)

    def deny(self, reason: str):
        """Denies the particular request.

        Args:
            reason (str): Reason for which the request has been denied.
        """
        api = APIRegistry.api_for(
            self.node_uid,
            self.syft_client_verify_key,
        )
        return api.services.request.undo(uid=self.id, reason=reason)

    def approve_with_client(self, client):
        print(f"Approving request for domain {client.name}")
        return client.api.services.request.apply(self.id)

    def apply(self, context: AuthedServiceContext) -> Result[SyftSuccess, SyftError]:
        change_context = ChangeContext.from_service(context)
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
        change_context = ChangeContext.from_service(context)
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

    def _get_latest_or_create_job(self) -> Union[Job, SyftError]:
        """Get the latest job for this requests user_code, or creates one if no jobs exist"""
        api = APIRegistry.api_for(self.node_uid, self.syft_client_verify_key)
        job_service = api.services.job

        existing_jobs = job_service.get_by_user_code_id(self.code.id)
        if isinstance(existing_jobs, SyftError):
            return existing_jobs

        if len(existing_jobs) == 0:
            job = job_service.create_job_for_user_code_id(self.code.id)
        else:
            job = existing_jobs[-1]

        return job

    def accept_by_depositing_result(self, result: Any, force: bool = False):
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
        else:
            # NOTE result is added at the end of function (once ActionObject is created)
            job_info = JobInfo(
                includes_metadata=True,
                includes_result=True,
                status=JobStatus.COMPLETED,
                resolved=True,
            )

        change = self.changes[0]
        if not change.is_type(UserCode):
            raise TypeError(
                f"accept_by_depositing_result can only be run on {UserCode} not "
                f"{change.linked_obj.object_type}"
            )
        if not type(change) == UserCodeStatusChange:
            raise TypeError(
                f"accept_by_depositing_result can only be run on {UserCodeStatusChange} not "
                f"{type(change)}"
            )

        api = APIRegistry.api_for(self.node_uid, self.syft_client_verify_key)
        if not api:
            raise Exception(f"Login to {self.node_uid} first.")

        is_approved = change.approved

        permission_request = self.approve(approve_nested=True)
        if isinstance(permission_request, SyftError):
            return permission_request

        code = change.linked_obj.resolve
        state = code.output_policy

        # This weird order is due to the fact that state is None before calling approve
        # we could fix it in a future release
        if is_approved:
            if not force:
                return SyftError(
                    message="Already approved, if you want to force updating the result use force=True"
                )
            action_obj_id = state.output_history[0].outputs[0]
            action_object = ActionObject.from_obj(
                result,
                id=action_obj_id,
                syft_client_verify_key=api.signing_key.verify_key,
                syft_node_location=api.node_uid,
            )
            blob_store_result = action_object._save_to_blob_storage()
            if isinstance(blob_store_result, SyftError):
                return blob_store_result
            result = api.services.action.set(action_object)
            if isinstance(result, SyftError):
                return result
        else:
            action_object = ActionObject.from_obj(
                result,
                syft_client_verify_key=api.signing_key.verify_key,
                syft_node_location=api.node_uid,
            )
            blob_store_result = action_object._save_to_blob_storage()
            if isinstance(blob_store_result, SyftError):
                return blob_store_result
            result = api.services.action.set(action_object)
            if isinstance(result, SyftError):
                return result

            ctx = AuthedServiceContext(credentials=api.signing_key.verify_key)

            state.apply_output(context=ctx, outputs=result)
            policy_state_mutation = ObjectMutation(
                linked_obj=change.linked_obj,
                attr_name="output_policy",
                match_type=True,
                value=state,
            )

            action_object_link = LinkedObject.from_obj(result, node_uid=self.node_uid)
            permission_change = ActionStoreChange(
                linked_obj=action_object_link,
                apply_permission_type=ActionPermission.READ,
            )

            new_changes = [policy_state_mutation, permission_change]
            result_request = api.services.request.add_changes(
                uid=self.id, changes=new_changes
            )
            if isinstance(result_request, SyftError):
                return result_request
            self = result_request

            approved = self.approve(disable_warnings=True, approve_nested=True)
            if isinstance(approved, SyftError):
                return approved

        job_info.result = action_object
        job = self._get_latest_or_create_job()
        job.apply_info(job_info)

        api = APIRegistry.api_for(self.node_uid, self.syft_client_verify_key)
        job_service = api.services.job
        res = job_service.update(job)
        if isinstance(res, SyftError):
            return res

        return SyftSuccess(message="Request submitted for updating result.")

    def sync_job(self, job_info: JobInfo, **kwargs) -> Result[SyftSuccess, SyftError]:
        if job_info.includes_result:
            return SyftError(
                message="This JobInfo includes a Result. Please use Request.accept_by_depositing_result instead."
            )

        api = APIRegistry.api_for(self.node_uid, self.syft_client_verify_key)
        job_service = api.services.job

        job = self._get_latest_or_create_job()
        job.apply_info(job_info)
        return job_service.update(job)


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

    name: Optional[str]


@serializable()
class SubmitRequest(SyftObject):
    __canonical_name__ = "SubmitRequest"
    __version__ = SYFT_OBJECT_VERSION_1

    changes: List[Change]
    requesting_user_verify_key: Optional[SyftVerifyKey]


def hash_changes(context: TransformContext) -> TransformContext:
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
    context.output["request_time"] = DateTime.now()
    return context


def check_requesting_user_verify_key(context: TransformContext) -> TransformContext:
    if context.obj.requesting_user_verify_key and context.node.is_root(
        context.credentials
    ):
        context.output[
            "requesting_user_verify_key"
        ] = context.obj.requesting_user_verify_key
    else:
        context.output["requesting_user_verify_key"] = context.credentials
    return context


def add_requesting_user_info(context: TransformContext) -> TransformContext:
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
def submit_request_to_request() -> List[Callable]:
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
    __version__ = SYFT_OBJECT_VERSION_1

    linked_obj: Optional[LinkedObject]
    attr_name: str
    value: Optional[Any]
    match_type: bool
    previous_value: Optional[Any]

    __repr_attrs__ = ["linked_obj", "attr_name"]

    def mutate(self, obj: Any, value: Optional[Any]) -> Any:
        # check if attribute is a property setter first
        # this seems necessary for pydantic types
        attr = getattr(type(obj), self.attr_name, None)
        if inspect.isdatadescriptor(attr):
            self.previous_value = attr.fget(obj)
            attr.fset(obj, value)

        else:
            self.previous_value = getattr(obj, self.attr_name, None)
            setattr(obj, self.attr_name, value)
        return obj

    def __repr_syft_nested__(self):
        return f"Mutate <b>{self.attr_name}</b> to <b>{self.value}</b>"

    def _run(
        self, context: ChangeContext, apply: bool
    ) -> Result[SyftSuccess, SyftError]:
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


def type_for_field(object_type: type, attr_name: str) -> Optional[type]:
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

    enum_type: Type[Enum]
    value: Optional[Enum]
    match_type: bool = True

    __repr_attrs__ = ["linked_obj", "attr_name", "value"]

    @property
    def valid(self) -> Union[SyftSuccess, SyftError]:
        if self.match_type and not isinstance(self.value, self.enum_type):
            return SyftError(
                message=f"{type(self.value)} must be of type: {self.enum_type}"
            )
        return SyftSuccess(message=f"{type(self)} valid")

    @staticmethod
    def from_obj(
        linked_obj: LinkedObject, attr_name: str, value: Optional[Enum] = None
    ) -> Self:
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
            obj = self.linked_obj.resolve_with_context(context)
            if obj.is_err():
                return SyftError(message=obj.err())
            obj = obj.ok()
            if apply:
                obj = self.mutate(obj)

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

    def __repr_syft_nested__(self):
        return f"Mutate <b>{self.enum_type}</b> to <b>{self.value}</b>"

    @property
    def link(self) -> Optional[SyftObject]:
        if self.linked_obj:
            return self.linked_obj.resolve
        return None


@serializable()
class UserCodeStatusChangeV1(Change):
    __canonical_name__ = "UserCodeStatusChange"
    __version__ = SYFT_OBJECT_VERSION_1

    value: UserCodeStatus
    linked_obj: LinkedObject
    match_type: bool = True
    __repr_attrs__ = [
        "link.service_func_name",
        "link.input_policy_type.__canonical_name__",
        "link.output_policy_type.__canonical_name__",
        "link.status.approved",
    ]


@serializable()
class UserCodeStatusChange(Change):
    __canonical_name__ = "UserCodeStatusChange"
    __version__ = SYFT_OBJECT_VERSION_2

    value: UserCodeStatus
    linked_obj: LinkedObject
    nested_solved: bool = False
    match_type: bool = True
    __repr_attrs__ = [
        "link.service_func_name",
        "link.input_policy_type.__canonical_name__",
        "link.output_policy_type.__canonical_name__",
        "link.worker_pool_name",
        "link.status.approved",
    ]

    @property
    def code(self):
        return self.link

    @property
    def codes(self):
        def recursive_code(node):
            codes = []
            for _, (obj, new_node) in node.items():
                codes.append(obj.resolve)
                codes.extend(recursive_code(new_node))
            return codes

        codes = [self.link]
        codes.extend(recursive_code(self.link.nested_codes))
        return codes

    def nested_repr(self, node=None, level=0):
        msg = ""
        if node is None:
            node = self.link.nested_codes
        for service_func_name, (_, new_node) in node.items():
            msg = "├──" + "──" * level + f"{service_func_name}<br>"
            msg += self.nested_repr(node=new_node, level=level + 1)
        return msg

    def __repr_syft_nested__(self):
        msg = f"Request to change <b>{self.link.service_func_name}</b> (Pool Id: <b>{self.link.worker_pool_name}</b>) "
        msg += "to permission <b>RequestStatus.APPROVED</b>"
        if self.nested_solved:
            if self.link.nested_codes == {}:
                msg += ". No nested requests"
            else:
                msg += ".<br><br>This change requests the following nested functions calls:<br>"
                msg += self.nested_repr()
        else:
            msg += ". Nested Requests not resolved"
        return msg

    def _repr_markdown_(self) -> str:
        link = self.link
        input_policy_type = (
            link.input_policy_type.__canonical_name__
            if link.input_policy_type is not None
            else None
        )
        output_policy_type = (
            link.output_policy_type.__canonical_name__
            if link.output_policy_type is not None
            else None
        )
        repr_dict = {
            "function": link.service_func_name,
            "input_policy_type": f"{input_policy_type}",
            "output_policy_type": f"{output_policy_type}",
            "approved": f"{link.status.approved}",
        }
        return markdown_as_class_with_fields(self, repr_dict)

    @property
    def approved(self) -> bool:
        return self.linked_obj.resolve.status.approved

    @property
    def valid(self) -> Union[SyftSuccess, SyftError]:
        if self.match_type and not isinstance(self.value, UserCodeStatus):
            return SyftError(
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

    def mutate(self, obj: UserCode, context: ChangeContext, undo: bool) -> Any:
        reason: str = context.extra_kwargs.get("reason", "")
        if not undo:
            res = obj.status.mutate(
                value=(self.value, reason),
                node_name=context.node.name,
                node_id=context.node.id,
                verify_key=context.node.signing_key.verify_key,
            )
            if isinstance(res, SyftError):
                return res
        else:
            res = obj.status.mutate(
                value=(UserCodeStatus.DENIED, reason),
                node_name=context.node.name,
                node_id=context.node.id,
                verify_key=context.node.signing_key.verify_key,
            )
        if not isinstance(res, SyftError):
            obj.status = res
            return obj
        return res

    def is_enclave_request(self, user_code: UserCode):
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
            obj = self.linked_obj.resolve_with_context(context)
            if obj.is_err():
                return Err(SyftError(message=obj.err()))
            obj = obj.ok()
            if apply:
                res = self.mutate(obj, context, undo=False)

                if isinstance(res, SyftError):
                    return Err(res.message)

                # relative
                from ..enclave.enclave_service import propagate_inputs_to_enclave

                user_code = res

                if self.is_enclave_request(user_code):
                    enclave_res = propagate_inputs_to_enclave(
                        user_code=res, context=context
                    )
                    if isinstance(enclave_res, SyftError):
                        return enclave_res
                self.linked_obj.update_with_context(context, user_code)
            else:
                res = self.mutate(obj, context, undo=True)
                if isinstance(res, SyftError):
                    return Err(res.message)

                # TODO: Handle Enclave approval.
                self.linked_obj.update_with_context(context, res)
            return Ok(SyftSuccess(message=f"{type(self)} Success"))
        except Exception as e:
            print(f"failed to apply {type(self)}. {e}")
            return Err(SyftError(message=str(e)))

    def apply(self, context: ChangeContext) -> Result[SyftSuccess, SyftError]:
        return self._run(context=context, apply=True)

    def undo(self, context: ChangeContext) -> Result[SyftSuccess, SyftError]:
        return self._run(context=context, apply=False)

    @property
    def link(self) -> Optional[SyftObject]:
        if self.linked_obj:
            return self.linked_obj.resolve
        return None


@migrate(UserCodeStatusChange, UserCodeStatusChangeV1)
def downgrade_usercodestatuschange_v2_to_v1():
    return [
        drop("nested_solved"),
    ]


@migrate(UserCodeStatusChangeV1, UserCodeStatusChange)
def upgrade_usercodestatuschange_v1_to_v2():
    return [
        make_set_default("nested_solved", True),
    ]
