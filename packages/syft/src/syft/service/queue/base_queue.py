# stdlib
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Mapping
from typing import Optional
from typing import Type

# relative
from ...serde.serializable import serializable


@serializable()
class QueueClientConfig:
    pass


class AbstractMessageHandler:
    queue: ClassVar[str]

    def message_handler(self):
        raise NotImplementedError


@serializable()
class QueueSubscriber:
    message_handler: Callable
    queue_name: str

    def receive(self):
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


@serializable()
class QueuePublisher:
    def send(self, queue_name: str, message: Any):
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class QueueClient:
    pass


@serializable()
class QueueConfig:
    """Base Store configuration

    Parameters:
        store_type: Type
            Document Store type
        client_config: Optional[StoreClientConfig]
            Backend-specific config
    """

    subscriber: Type[QueueSubscriber]
    publisher: Type[QueuePublisher]
    client_config: Optional[QueueClientConfig]
    client_type: Type[QueueClient]


@serializable()
class BaseQueueRouter:
    config: QueueConfig
    subscribers: Mapping[str, list[QueueSubscriber]]

    def __init__(self, config: QueueConfig):
        self.config = config
        self.post_init()

    def post_init(self) -> None:
        pass

    def start(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> "BaseQueueRouter":
        self.start()
        return self

    def close(self) -> None:
        raise NotImplementedError

    def create_subscriber(
        self, message_handler: AbstractMessageHandler
    ) -> QueueSubscriber:
        raise NotImplementedError

    @property
    def publisher(self) -> QueuePublisher:
        raise NotImplementedError
