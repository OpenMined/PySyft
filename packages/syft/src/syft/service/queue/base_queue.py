# stdlib
from typing import Optional
from typing import Type

# relative
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftBaseObject


@serializable
class QueueClientConfig():
    name: str


@serializable
class QueueRouter():
    config: QueueClientConfig

    def start()

    def close(self)

    def subscriber() -> QueueSubscriber[T]:
        pass

    def publisher() -> QueuePublisher[T]:
        pass


@serializable
class QueueSubscriber():
    def connect():
        raise NotImplementedError

    def receive(self, message):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def close(self):
        raise


@serializable
class QueuePublisher():
    def send(self, message):
        raise NotImplementedError


@serializable()
class QueueConfig(SyftBaseObject):
    """Base Store configuration

    Parameters:
        store_type: Type
            Document Store type
        client_config: Optional[StoreClientConfig]
            Backend-specific config
    """

    __canonical_name__ = "QueueConfig"
    __version__ = SYFT_OBJECT_VERSION_1

    queue_type: Type[QueueClient]
    client_config: Optional[QueueClientConfig]


@serializable
class Queue():
    """handle syft related stuff"""
    config: QueueConfig
