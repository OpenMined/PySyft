# stdlib
from typing import ClassVar

# third party
from pydantic import model_validator
from result import Err
from result import Ok
from result import Result

# relative
from ...client.api import APIRegistry
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.linked_obj import LinkedObject
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syncable_object import SyncableSyftObject
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_object import ActionObject
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


@serializable()
class ExecutionOutput(SyncableSyftObject):
    __canonical_name__ = "ExecutionOutput"
    __version__ = SYFT_OBJECT_VERSION_2

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
        node_uid: UID,
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
            node_uid=node_uid,
        )

        if job_id:
            job_link = LinkedObject.from_uid(
                object_uid=job_id,
                object_type=Job,
                service_type=JobService,
                node_uid=node_uid,
            )
        else:
            job_link = None
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
        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            raise ValueError(
                f"Can't access the api. Please log in to {self.syft_node_location}"
            )
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


@instrument
@serializable()
class OutputStash(BaseUIDStoreStash):
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
        qks = QueryKeys(
            qks=[UserCodeIdPartitionKey.with_obj(user_code_id)],
        )
        return self.query_all(
            credentials=credentials, qks=qks, order_by=CreatedAtPartitionKey
        )

    def get_by_job_id(
        self, credentials: SyftVerifyKey, user_code_id: UID
    ) -> Result[ExecutionOutput | None, str]:
        qks = QueryKeys(
            qks=[JobIdPartitionKey.with_obj(user_code_id)],
        )
        res = self.query_all(
            credentials=credentials, qks=qks, order_by=CreatedAtPartitionKey
        )
        if res.is_err():
            return res
        else:
            res = res.ok()
            if len(res) == 0:
                return Ok(None)
            elif len(res) > 1:
                return Err(message="Too many outputs found")
            else:
                return Ok(res[0])

    def get_by_output_policy_id(
        self, credentials: SyftVerifyKey, output_policy_id: UID
    ) -> Result[list[ExecutionOutput], str]:
        qks = QueryKeys(
            qks=[OutputPolicyIdPartitionKey.with_obj(output_policy_id)],
        )
        return self.query_all(
            credentials=credentials, qks=qks, order_by=CreatedAtPartitionKey
        )


@instrument
@serializable()
class OutputService(AbstractService):
    store: DocumentStore
    stash: OutputStash

    def __init__(self, store: DocumentStore):
        self.store = store
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
    ) -> ExecutionOutput | SyftError:
        output = ExecutionOutput.from_ids(
            output_ids=output_ids,
            user_code_id=user_code_id,
            executing_user_verify_key=executing_user_verify_key,
            node_uid=context.node.id,  # type: ignore
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
            credentials=context.node.verify_key,  # type: ignore
            user_code_id=user_code_id,
        )
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(
        path="output.get_by_job_id",
        name="get_by_job_id",
        roles=ADMIN_ROLE_LEVEL,
    )
    def get_by_job_id(
        self, context: AuthedServiceContext, user_code_id: UID
    ) -> ExecutionOutput | None | SyftError:
        result = self.stash.get_by_job_id(
            credentials=context.node.verify_key,  # type: ignore
            user_code_id=user_code_id,
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
            credentials=context.node.verify_key,  # type: ignore
            output_policy_id=output_policy_id,  # type: ignore
        )
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
