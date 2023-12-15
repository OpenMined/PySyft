# stdlib
from typing import Union

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseStash, PartitionKey
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..response import SyftError
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL


@serializable()
class ConsumerItem(SyftObject):
    __canonical_name__ = "QueueConsumerItem"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_searchable__ = ["consumer_id", "worker_id", "job_id"]
    __repr_attrs__ = ["consumer_id", "worker_id", "job_id"]

    consumer_id: str
    worker_id: str
    job_id: str


@instrument
@serializable()
class ConsumerStash(BaseStash):
    object_type = ConsumerItem
    settings: PartitionSettings = PartitionSettings(
        name=ConsumerItem.__canonical_name__, object_type=ConsumerItem
    )

    def get_by_consumer_id(self, credentials: SyftVerifyKey, consumer_id: str):
        qks = QueryKeys(
            qks=[PartitionKey(key="consumer_id", type_=str).with_obj(consumer_id)]
        )
        item = self.query_one(credentials=credentials, qks=qks)
        return item


@instrument
@serializable()
class ConsumerService(AbstractService):
    store: DocumentStore
    stash: ConsumerStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = ConsumerStash(store=store)

    @service_method(
        path="consumer.get_consumers",
        name="get_consumers",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_consumers(
        self, context: AuthedServiceContext
    ) -> Union[list[ConsumerItem], SyftError]:
        res = self.stash.get_all(context.credentials)
        if res.is_err():
            return SyftError(message=res.err())
        else:
            return res.ok()
