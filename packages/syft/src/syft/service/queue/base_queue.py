# stdlib
from typing import Any
from typing import ClassVar
from typing import TYPE_CHECKING

# relative
from ...serde.serializable import serializable
from ...service.context import AuthedServiceContext
from ...types.uid import UID
from ..response import SyftSuccess
from ..worker.worker_stash import WorkerStash

if TYPE_CHECKING:
    # relative
    from .queue_stash import QueueStash


@serializable(canonical_name="QueueClientConfig", version=1)
class QueueClientConfig:
    pass


@serializable(canonical_name="AbstractMessageHandler", version=1)
class AbstractMessageHandler:
    queue_name: ClassVar[str]

    @staticmethod
    def handle_message(message: bytes, syft_worker_id: UID) -> None:
        raise NotImplementedError


@serializable(
    attrs=["message_handler", "queue_name", "address"],
    canonical_name="QueueConsumer",
    version=1,
)
class QueueConsumer:
    message_handler: AbstractMessageHandler
    queue_name: str
    address: str

    def receive(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


@serializable(canonical_name="QueueProducer", version=1)
class QueueProducer:
    queue_name: str

    @property
    def address(self) -> str:
        raise NotImplementedError

    def send(
        self,
        worker: bytes,
        message: Any,
    ) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


@serializable(canonical_name="QueueClient", version=1)
class QueueClient:
    def __init__(self, config: QueueClientConfig) -> None:
        raise NotImplementedError


@serializable(canonical_name="QueueConfig", version=1)
class QueueConfig:
    """Base Queue configuration"""

    client_type: type[QueueClient]
    client_config: QueueClientConfig


@serializable(canonical_name="BaseQueueManager", version=1)
class BaseQueueManager:
    config: QueueConfig

    def __init__(self, config: QueueConfig):
        self.config = config
        self.post_init()

    def post_init(self) -> None:
        pass

    def close(self) -> SyftSuccess:
        raise NotImplementedError

    def create_consumer(
        self,
        message_handler: type[AbstractMessageHandler],
        service_name: str,
        worker_stash: WorkerStash | None = None,
        address: str | None = None,
        syft_worker_id: UID | None = None,
    ) -> QueueConsumer:
        raise NotImplementedError

    def create_producer(
        self,
        queue_name: str,
        queue_stash: "QueueStash",
        context: AuthedServiceContext,
        worker_stash: WorkerStash,
    ) -> QueueProducer:
        raise NotImplementedError

    def send(self, message: bytes, queue_name: str) -> SyftSuccess:
        raise NotImplementedError

    @property
    def publisher(self) -> QueueProducer:
        raise NotImplementedError
