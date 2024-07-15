# stdlib
from typing import Any

# relative
from ...serde.serializable import serializable
from ...service.response import SyftError
from ...service.response import SyftSuccess
from ...service.user.user_roles import GUEST_ROLE_LEVEL
from ...store.document_store import DocumentStore
from ...types.twin_object import TwinObject
from ...types.uid import UID
from ..action.action_object import ActionObject
from ..code.user_code import SubmitUserCode
from ..code.user_code import UserCode
from ..code.user_code import UserCodeStatus
from ..context import AuthedServiceContext
from ..model.model import ModelRef
from ..service import AbstractService
from ..service import service_method


@serializable()
class EnclaveService(AbstractService):
    """Contains service methods exposed by Enclaves."""

    store: DocumentStore

    def __init__(self, store: DocumentStore) -> None:
        self.store = store

    @service_method(
        path="enclave.setup_enclave",
        name="setup_enclave",
        roles=GUEST_ROLE_LEVEL,  # TODO 🟣 Only an enclave's owner domain node should call this
    )
    def setup_enclave(
        self, context: AuthedServiceContext, code: UserCode | SubmitUserCode
    ) -> SyftSuccess | SyftError:
        if not context.node or not context.node.signing_key:
            return SyftError(message=f"{type(context)} has no node")

        root_context = context.as_root_context()

        # TODO add queuing mechanism

        if isinstance(code, UserCode):
            code = code.to(SubmitUserCode)

        result = context.node.get_service("usercodeservice").submit(root_context, code)
        if isinstance(result, SyftError):
            return result
        return SyftSuccess(message="Enclave setup successful")

    @service_method(
        path="enclave.upload_assets",
        name="upload_assets",
        roles=GUEST_ROLE_LEVEL,
    )
    def upload_assets(
        self,
        context: AuthedServiceContext,
        user_code_id: UID,
        action_objects: list[ActionObject] | list[TwinObject],
    ) -> SyftSuccess | SyftError:
        if not context.node or not context.node.signing_key:
            return SyftError(message=f"{type(context)} has no node")

        root_context = context.as_root_context()

        code_service = context.node.get_service("usercodeservice")
        action_service = context.node.get_service("actionservice")

        # Get the code
        code: UserCode = code_service.get_by_uid(context=root_context, uid=user_code_id)

        init_kwargs = code.input_policy_init_kwargs
        if not code or not init_kwargs:
            return SyftError(message="No assets to transfer")

        node_identity_map = {node.verify_key: node for node in init_kwargs.keys()}
        uploading_domain_identity = node_identity_map.get(context.credentials)

        if not uploading_domain_identity:
            return SyftError(
                message="You are not allowed to upload assets for the given code"
            )

        kwargs_for_uploading_domain = init_kwargs[uploading_domain_identity]

        for action_object in action_objects:
            if action_object.id not in kwargs_for_uploading_domain.values():
                return SyftError(
                    message=f"You are not allowed to upload the asset with id '{action_object.id}'"
                )

        pending_assets_for_uploading_domain = set(kwargs_for_uploading_domain.values())
        for action_object in action_objects:
            if type(action_object) == ModelRef:
                result = action_object.store_ref_objs_to_store(
                    context=root_context, clear_ref_objs=True
                )
            else:
                result = action_service.set(root_context, action_object)
            if isinstance(result, SyftError):
                # TODO 🟣 Rollback previously uploaded assets if any error occurs
                return result
            pending_assets_for_uploading_domain.remove(action_object.id)

        # Let's approve the code
        if len(pending_assets_for_uploading_domain) == 0:
            approved_status_with_reason = (
                UserCodeStatus.APPROVED,
                "All dependent assets uploaded by this domain node.",
            )
            status = code.get_status(root_context)
            status.status_dict[uploading_domain_identity] = approved_status_with_reason
            status_link = code.status_link
            if not status_link:
                return SyftError(
                    message=f"Code '{code.service_func_name}' does not have a status link."
                )
            res = status_link.update_with_context(root_context, status)
            if isinstance(res, SyftError):
                return res

        return SyftSuccess(
            message=f"{len(action_objects)} assets uploaded successfully"
        )

    @service_method(
        path="enclave.execute_code",
        name="execute_code",
        roles=GUEST_ROLE_LEVEL,
    )
    def execute_code(self, context: AuthedServiceContext, user_code_id: UID) -> Any:
        if not context.node or not context.node.signing_key:
            return SyftError(message=f"{type(context)} has no node")

        # TODO only allow execution for domain nodes in output_policy.share_result_with list
        root_context = context.as_root_context()

        code_service = context.node.get_service("usercodeservice")
        job_service = context.node.get_service("jobservice")

        code: UserCode = code_service.get_by_uid(context=root_context, uid=user_code_id)

        jobs = job_service.get_by_user_code_id(
            context=root_context, user_code_id=code.id
        )
        if jobs:
            job = jobs[-1]
            return job.wait().get()

        init_kwargs = (
            code.input_policy_init_kwargs.values()
            if code.input_policy_init_kwargs is not None
            else []
        )
        kwargs = {k: v for d in init_kwargs for k, v in d.items()}

        admin_client = context.node.root_client
        job = admin_client.api.services.code.call(blocking=False, uid=code.id, **kwargs)
        execution_result = job.wait().get()
        result = get_encrypted_result(context, execution_result)
        return result


def get_encrypted_result(context: AuthedServiceContext, result: Any) -> Any:
    # TODO 🟣 Encrypt the result before sending it back to the user
    return result
