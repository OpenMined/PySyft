# stdlib
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.linked_obj import LinkedObject
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_object import ActionObject
from ..context import AuthedServiceContext
from ..response import SyftError
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import GUEST_ROLE_LEVEL

CreatedAtPartitionKey = PartitionKey(key="created_at", type_=DateTime)
UserCodeIdPartitionKey = PartitionKey(key="user_code_id", type_=UID)


class ExecutionOutput(SyftObject):
    __canonical_name__ = "ExecutionOutput"
    __version__ = SYFT_OBJECT_VERSION_1

    executing_user_verify_key: SyftVerifyKey
    user_code_link: LinkedObject
    output_links: Optional[Union[List[LinkedObject], Dict[str, LinkedObject]]]
    job_link: Optional[LinkedObject] = None
    created_at: DateTime = DateTime.now()

    __attr_searchable__ = ["user_code_id"]
    __repr_attrs__ = ["user_code_id", "job_id", "output_ids"]

    @classmethod
    def from_output_ids(
        cls: Type["ExecutionOutput"],
        output_ids: Union[UID, List[UID], Dict[str, UID]],
        node_uid: UID,
    ) -> "ExecutionOutput":
        # relative
        from ..action.action_service import ActionService

        def make_link(uid: UID) -> LinkedObject:
            return LinkedObject.from_uid(
                object_uid=uid,
                object_type=ActionObject,
                service_type=ActionService,
                node_uid=node_uid,
            )

        if isinstance(output_ids, dict):
            output_links = {k: make_link(v) for k, v in output_ids.items()}
        elif isinstance(output_ids, UID):
            output_links = [make_link(output_ids)]
        else:
            output_links = [make_link(x) for x in output_ids]
        return cls(output_links=output_links)

    @property
    def outputs(self) -> Union[List[ActionObject], Dict[str, ActionObject]]:
        if isinstance(self.output_links, dict):
            return {k: v.resolve for k, v in self.output_links.items()}
        else:
            return [x.resolve for x in self.output_links]

    @property
    def output_ids(self) -> Union[List[UID], Dict[str, UID]]:
        if isinstance(self.output_links, dict):
            return {k: v.object_uid for k, v in self.output_links.items()}
        else:
            return [x.object_uid for x in self.output_links]

    @property
    def user_code_id(self) -> UID:
        return self.user_code_link.object_uid

    @property
    def job_id(self) -> Optional[UID]:
        return self.job_link.object_uid if self.job_link else None


@instrument
@serializable()
class OutputStash(BaseUIDStoreStash):
    object_type = ExecutionOutput
    settings: PartitionSettings = PartitionSettings(
        name=ExecutionOutput.__canonical_name__, object_type=ExecutionOutput
    )

    def __init__(self, store):
        super().__init__(store)
        self.store = store
        self.settings = self.settings
        self._object_type = self.object_type

    def get_by_user_code_id(
        self, credentials: SyftVerifyKey, user_code_id: UID
    ) -> Union[List[ExecutionOutput], SyftError]:
        qks = QueryKeys(
            partition_keys=[UserCodeIdPartitionKey.with_obj(user_code_id)],
        )

        return self.query_all(
            credentials=credentials, qks=qks, order_by=CreatedAtPartitionKey
        )


class OutputService(AbstractService):
    store: DocumentStore
    stash: OutputStash

    def __init__(self, store: DocumentStore):
        self.store = store
        self.stash = OutputStash(store=store)

    @service_method(
        path="result.get_by_user_code_id",
        name="get_by_user_code_id",
        role=GUEST_ROLE_LEVEL,
    )
    def get_by_user_code_id(
        self, context: AuthedServiceContext, user_code_id: UID
    ) -> Union[List[ExecutionOutput], SyftError]:
        return self.stash.get_by_user_code_id(
            credentials=context.node.verify_key, user_code_id=user_code_id
        )
