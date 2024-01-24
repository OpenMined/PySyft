# stdlib
from typing import Any
from typing import ClassVar
from typing import Optional
from typing import Type
from typing import Union

# relative
from ...serde.serializable import serializable
from ..response import SyftError
from ..response import SyftSuccess


@serializable()
class QueueClientConfig:
    pass


@serializable()
class AbstractMessageHandler:
    queue_name: ClassVar[str]

    @staticmethod
    def handle_message(message: bytes):
        raise NotImplementedError


@serializable(attrs=["message_handler", "queue_name", "address"])
class QueueConsumer:
    message_handler: AbstractMessageHandler
    queue_name: str
    address: str

    def receive(self):
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


@serializable()
class QueueProducer:
    address: str
    queue_name: str

    def send(
        self,
        message: Any,
    ):
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


@serializable()
class QueueClient:
    pass


@serializable()
class QueueConfig:
    """Base Queue configuration"""

    client_config: Type[QueueClientConfig]
    client_type: Type[QueueClient]


@serializable()
class BaseQueueManager:
    config: QueueConfig

    def __init__(self, config: QueueConfig):
        self.config = config
        self.post_init()

    def post_init(self) -> None:
        pass

    def close(self) -> None:
        raise NotImplementedError

    def create_consumer(
        self,
        message_handler: Type[AbstractMessageHandler],
        address: Optional[str],
    ) -> QueueConsumer:
        raise NotImplementedError

    def create_producer(self, queue_name: str) -> QueueProducer:
        raise NotImplementedError

    def send(self, message: bytes, queue_name: str) -> Union[SyftSuccess, SyftError]:
        raise NotImplementedError

    @property
    def publisher(self) -> QueueProducer:
        raise NotImplementedError
