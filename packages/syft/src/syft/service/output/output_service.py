# stdlib
from typing import ClassVar

# third party
from pydantic import model_validator

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.db import DBManager
from ...store.db.stash import ObjectStash
from ...store.document_store_errors import StashException
from ...store.linked_obj import LinkedObject
from ...types.datetime import DateTime
from ...types.result import as_result
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syncable_object import SyncableSyftObject
from ...types.uid import UID
from ..action.action_object import ActionObject
from ..action.action_permissions import ActionObjectREAD
from ..context import AuthedServiceContext
from ..service import AbstractService
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from ..user.user_roles import GUEST_ROLE_LEVEL


@serializable()
class ExecutionOutput(SyncableSyftObject):
    __canonical_name__ = "ExecutionOutput"
    __version__ = SYFT_OBJECT_VERSION_1

    executing_user_verify_key: SyftVerifyKey
    user_code_link: LinkedObject
    output_ids: list[UID] | dict[str, UID] | None = None
    job_link: LinkedObject | None = None
    created_at: DateTime = DateTime.now()
    input_ids: dict[str, UID] | None = None

    # Required for __attr_searchable__, set by model_validator
    user_code_id: UID
    job_id: UID | None = None

    # Output policy is not a linked object because its saved on the usercode
    output_policy_id: UID | None = None

    __attr_searchable__: ClassVar[list[str]] = [
        "user_code_id",
        "created_at",
        "output_policy_id",
        "job_id",
    ]
    __repr_attrs__: ClassVar[list[str]] = [
        "created_at",
        "user_code_id",
        "job_id",
        "output_ids",
    ]

    @model_validator(mode="before")
    @classmethod
    def add_searchable_link_ids(cls, values: dict) -> dict:
        if "user_code_link" in values:
            values["user_code_id"] = values["user_code_link"].object_uid
        if values.get("job_link"):
            values["job_id"] = values["job_link"].object_uid
        return values

    @classmethod
    def from_ids(
        cls: type["ExecutionOutput"],
        output_ids: UID | list[UID] | dict[str, UID],
        user_code_id: UID,
        executing_user_verify_key: SyftVerifyKey,
        server_uid: UID,
        job_id: UID | None = None,
        output_policy_id: UID | None = None,
        input_ids: dict[str, UID] | None = None,
    ) -> "ExecutionOutput":
        # relative
        from ..code.user_code_service import UserCode
        from ..code.user_code_service import UserCodeService
        from ..job.job_service import Job
        from ..job.job_service import JobService

        if isinstance(output_ids, UID):
            output_ids = [output_ids]

        user_code_link = LinkedObject.from_uid(
            object_uid=user_code_id,
            object_type=UserCode,
            service_type=UserCodeService,
            server_uid=server_uid,
        )

        if job_id:
            job_link = LinkedObject.from_uid(
                object_uid=job_id,
                object_type=Job,
                service_type=JobService,
                server_uid=server_uid,
            )
        else:
            job_link = None

        if input_ids is not None:
            input_ids = {k: v for k, v in input_ids.items() if isinstance(v, UID)}
        return cls(
            output_ids=output_ids,
            user_code_link=user_code_link,
            job_link=job_link,
            executing_user_verify_key=executing_user_verify_key,
            output_policy_id=output_policy_id,
            input_ids=input_ids,
        )

    @property
    def outputs(self) -> list[ActionObject] | dict[str, ActionObject] | None:
        api = self.get_api()
        action_service = api.services.action

        # TODO: error handling for action_service.get
        if isinstance(self.output_ids, dict):
            return {k: action_service.get(v) for k, v in self.output_ids.items()}
        elif isinstance(self.output_ids, list):
            return [action_service.get(v) for v in self.output_ids]
        else:
            return None

    @property
    def output_id_list(self) -> list[UID]:
        ids = self.output_ids
        if isinstance(ids, dict):
            return list(ids.values())
        elif isinstance(ids, list):
            return ids
        return []

    @property
    def input_id_list(self) -> list[UID]:
        ids = self.input_ids
        if isinstance(ids, dict):
            return list(ids.values())
        return []

    def check_input_ids(self, kwargs: dict[str, UID]) -> bool:
        """
        Checks the input IDs against the stored input IDs.

        Args:
            kwargs (dict[str, UID]): A dictionary containing the input IDs to be checked.

        Returns:
            bool: True if the input IDs are valid, False otherwise.
        """
        if not self.input_ids:
            return True
        for key, value in kwargs.items():  # Iterate over items of kwargs dictionary
            if key not in self.input_ids or self.input_ids[key] != value:
                return False
        return True

    def get_sync_dependencies(self, context: AuthedServiceContext) -> list[UID]:
        # Output ids, user code id, job id
        res = []

        res.extend(self.output_id_list)
        res.append(self.user_code_id)
        if self.job_id:
            res.append(self.job_id)

        return res


@serializable(canonical_name="OutputStashSQL", version=1)
class OutputStash(ObjectStash[ExecutionOutput]):
    @as_result(StashException)
    def get_by_user_code_id(
        self, credentials: SyftVerifyKey, user_code_id: UID
    ) -> list[ExecutionOutput]:
        return self.get_all(
            credentials=credentials,
            filters={"user_code_id": user_code_id},
        ).unwrap()

    @as_result(StashException)
    def get_by_job_id(
        self, credentials: SyftVerifyKey, job_id: UID
    ) -> ExecutionOutput | None:
        return self.get_one(
            credentials=credentials,
            filters={"job_id": job_id},
        ).unwrap()

    @as_result(StashException)
    def get_by_output_policy_id(
        self, credentials: SyftVerifyKey, output_policy_id: UID
    ) -> list[ExecutionOutput]:
        return self.get_all(
            credentials=credentials,
            filters={"output_policy_id": output_policy_id},
        ).unwrap()


@serializable(canonical_name="OutputService", version=1)
class OutputService(AbstractService):
    stash: OutputStash

    def __init__(self, store: DBManager):
        self.stash = OutputStash(store=store)

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
    ) -> ExecutionOutput:
        output = ExecutionOutput.from_ids(
            output_ids=output_ids,
            user_code_id=user_code_id,
            executing_user_verify_key=executing_user_verify_key,
            server_uid=context.server.id,  # type: ignore
            job_id=job_id,
            output_policy_id=output_policy_id,
            input_ids=input_ids,
        )

        return self.stash.set(context.credentials, output).unwrap()

    @service_method(
        path="output.get_by_user_code_id",
        name="get_by_user_code_id",
        roles=GUEST_ROLE_LEVEL,
    )
    def get_by_user_code_id(
        self, context: AuthedServiceContext, user_code_id: UID
    ) -> list[ExecutionOutput]:
        return self.stash.get_by_user_code_id(
            credentials=context.server.verify_key,  # type: ignore
            user_code_id=user_code_id,
        ).unwrap()

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
    ) -> bool:
        all_outputs = self.get_by_user_code_id(context, user_code_id)
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
            if context.server.services.action.stash.has_permissions(permissions):
                return True

        return False

    @service_method(
        path="output.get_by_job_id",
        name="get_by_job_id",
        roles=ADMIN_ROLE_LEVEL,
    )
    def get_by_job_id(
        self, context: AuthedServiceContext, job_id: UID
    ) -> ExecutionOutput:
        return self.stash.get_by_job_id(
            credentials=context.server.verify_key,  # type: ignore
            job_id=job_id,
        ).unwrap()

    @service_method(
        path="output.get_by_output_policy_id",
        name="get_by_output_policy_id",
        roles=GUEST_ROLE_LEVEL,
    )
    def get_by_output_policy_id(
        self, context: AuthedServiceContext, output_policy_id: UID
    ) -> list[ExecutionOutput]:
        return self.stash.get_by_output_policy_id(
            credentials=context.server.verify_key,  # type: ignore
            output_policy_id=output_policy_id,  # type: ignore
        ).unwrap()

    @service_method(
        path="output.get",
        name="get",
        roles=GUEST_ROLE_LEVEL,
    )
    def get(self, context: AuthedServiceContext, id: UID) -> ExecutionOutput:
        return self.stash.get_by_uid(context.credentials, id).unwrap()

    @service_method(path="output.get_all", name="get_all", roles=GUEST_ROLE_LEVEL)
    def get_all(self, context: AuthedServiceContext) -> list[ExecutionOutput]:
        return self.stash.get_all(context.credentials).unwrap()


TYPE_TO_SERVICE[ExecutionOutput] = OutputService
