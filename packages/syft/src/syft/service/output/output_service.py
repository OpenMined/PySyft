# stdlib

# third party
from result import Result
from syft.service.job.job_sql_stash import ObjectStash
from syft.service.output.execution_output import ExecutionOutput
from syft.service.output.execution_output_sql import ExecutionOutputDB

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...types.datetime import DateTime
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_permissions import ActionObjectREAD
from ..context import AuthedServiceContext
from ..response import SyftError
from ..service import AbstractService
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from ..user.user_roles import GUEST_ROLE_LEVEL

CreatedAtPartitionKey = PartitionKey(key="created_at", type_=DateTime)
UserCodeIdPartitionKey = PartitionKey(key="user_code_id", type_=UID)
JobIdPartitionKey = PartitionKey(key="job_id", type_=UID)
OutputPolicyIdPartitionKey = PartitionKey(key="output_policy_id", type_=UID)


@instrument
@serializable(canonical_name="OutputStashSQL", version=1)
class OutputStashSQL(ObjectStash[ExecutionOutput, ExecutionOutputDB]):
    object_type = ExecutionOutput
    settings: PartitionSettings = PartitionSettings(
        name=ExecutionOutput.__canonical_name__, object_type=ExecutionOutput
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store)
        self.store = store
        self.settings = self.settings
        self._object_type = self.object_type

    def get_by_user_code_id(
        self, credentials: SyftVerifyKey, user_code_id: UID
    ) -> Result[list[ExecutionOutput], str]:
        return self.get_one_by_property(
            credentials=credentials,
            property_name="user_code_id",
            property_value=user_code_id,
        )

    def get_by_job_id(
        self, credentials: SyftVerifyKey, job_id: UID
    ) -> Result[ExecutionOutput | None, str]:
        return self.get_one_by_property(
            credentials=credentials, property_name="job_id", property_value=job_id
        )

    def get_by_output_policy_id(
        self, credentials: SyftVerifyKey, output_policy_id: UID
    ) -> Result[list[ExecutionOutput], str]:
        return self.get_one_by_property(
            credentials=credentials,
            property_name="output_policy_id",
            property_value=output_policy_id,
        )


@instrument
@serializable(canonical_name="OutputService", version=1)
class OutputService(AbstractService):
    store: DocumentStore
    stash: OutputStashSQL

    def __init__(self, store: DocumentStore):
        self.store = store
        self.stash = OutputStashSQL(store=store)

    @service_method(
        path="output.create",
        name="create",
        roles=GUEST_ROLE_LEVEL,
    )
    def create(
        self,
        context: AuthedServiceContext,
        user_code_id: UID,
        output_ids: UID | list[UID] | dict[str, UID],
        executing_user_verify_key: SyftVerifyKey,
        job_id: UID | None = None,
        output_policy_id: UID | None = None,
        input_ids: dict[str, UID] | None = None,
    ) -> ExecutionOutput | SyftError:
        output = ExecutionOutput.from_ids(
            output_ids=output_ids,
            user_code_id=user_code_id,
            executing_user_verify_key=executing_user_verify_key,
            server_uid=context.server.id,  # type: ignore
            job_id=job_id,
            output_policy_id=output_policy_id,
            input_ids=input_ids,
        )

        res = self.stash.set(context.credentials, output)
        return res

    @service_method(
        path="output.get_by_user_code_id",
        name="get_by_user_code_id",
        roles=GUEST_ROLE_LEVEL,
    )
    def get_by_user_code_id(
        self, context: AuthedServiceContext, user_code_id: UID
    ) -> list[ExecutionOutput] | SyftError:
        result = self.stash.get_by_user_code_id(
            credentials=context.server.verify_key,  # type: ignore
            user_code_id=user_code_id,
        )
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(
        path="output.has_output_read_permissions",
        name="has_output_read_permissions",
        roles=GUEST_ROLE_LEVEL,
    )
    def has_output_read_permissions(
        self,
        context: AuthedServiceContext,
        user_code_id: UID,
        user_verify_key: SyftVerifyKey,
    ) -> bool | SyftError:
        action_service = context.server.get_service("actionservice")
        all_outputs = self.get_by_user_code_id(context, user_code_id)
        if isinstance(all_outputs, SyftError):
            return all_outputs
        for output in all_outputs:
            # TODO tech debt: unclear why code owner can see outputhistory without permissions.
            # It is not a security issue (output history has no data) it is confusing for user
            # if not self.stash.has_permission(
            #     ActionObjectREAD(uid=output.id, credentials=user_verify_key)
            # ):
            #     continue

            # Check if all output ActionObjects have permissions
            result_ids = output.output_id_list
            permissions = [
                ActionObjectREAD(uid=_id.id, credentials=user_verify_key)
                for _id in result_ids
            ]
            if action_service.store.has_permissions(permissions):
                return True

        return False

    @service_method(
        path="output.get_by_job_id",
        name="get_by_job_id",
        roles=ADMIN_ROLE_LEVEL,
    )
    def get_by_job_id(
        self, context: AuthedServiceContext, job_id: UID
    ) -> ExecutionOutput | None | SyftError:
        result = self.stash.get_by_job_id(
            credentials=context.server.verify_key,  # type: ignore
            job_id=job_id,
        )
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(
        path="output.get_by_output_policy_id",
        name="get_by_output_policy_id",
        roles=GUEST_ROLE_LEVEL,
    )
    def get_by_output_policy_id(
        self, context: AuthedServiceContext, output_policy_id: UID
    ) -> list[ExecutionOutput] | SyftError:
        result = self.stash.get_by_output_policy_id(
            credentials=context.server.verify_key,  # type: ignore
            output_policy_id=output_policy_id,  # type: ignore
        )
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(
        path="output.get",
        name="get",
        roles=GUEST_ROLE_LEVEL,
    )
    def get(
        self, context: AuthedServiceContext, id: UID
    ) -> ExecutionOutput | SyftError:
        result = self.stash.get_by_uid(context.credentials, id)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(path="output.get_all", name="get_all", roles=GUEST_ROLE_LEVEL)
    def get_all(
        self, context: AuthedServiceContext
    ) -> list[ExecutionOutput] | SyftError:
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())


TYPE_TO_SERVICE[ExecutionOutput] = OutputService
