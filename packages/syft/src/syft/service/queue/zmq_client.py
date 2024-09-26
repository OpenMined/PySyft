# stdlib
from collections import defaultdict
import logging
import socketserver

# relative
from ...serde.serializable import serializable
from ...service.context import AuthedServiceContext
from ...types.errors import SyftException
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ...util.util import get_queue_address
from ..response import SyftSuccess
from ..worker.worker_stash import WorkerStash
from .base_queue import AbstractMessageHandler
from .base_queue import QueueClient
from .base_queue import QueueClientConfig
from .base_queue import QueueConfig
from .queue import ConsumerType
from .queue_stash import QueueStash
from .zmq_consumer import ZMQConsumer
from .zmq_producer import ZMQProducer

logger = logging.getLogger(__name__)


@serializable()
class ZMQClientConfig(SyftObject, QueueClientConfig):
    __canonical_name__ = "ZMQClientConfig"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID | None = None  # type: ignore[assignment]
    hostname: str = "127.0.0.1"
    queue_port: int | None = None
    # TODO: setting this to false until we can fix the ZMQ
    # port issue causing tests to randomly fail
    create_producer: bool = False
    n_consumers: int = 0
    consumer_service: str | None = None


@serializable(attrs=["host"], canonical_name="ZMQClient", version=1)
class ZMQClient(QueueClient):
    """ZMQ Client for creating producers and consumers."""

    producers: dict[str, ZMQProducer]
    consumers: defaultdict[str, list[ZMQConsumer]]

    def __init__(self, config: ZMQClientConfig) -> None:
        self.host = config.hostname
        self.producers = {}
        self.consumers = defaultdict(list)
        self.config = config

    @staticmethod
    def _get_free_tcp_port(host: str) -> int:
        with socketserver.TCPServer((host, 0), None) as s:
            free_port = s.server_address[1]

        return free_port

    def add_producer(
        self,
        queue_name: str,
        port: int | None = None,
        queue_stash: QueueStash | None = None,
        worker_stash: WorkerStash | None = None,
        context: AuthedServiceContext | None = None,
    ) -> ZMQProducer:
        """Add a producer of a queue.

        A queue can have at most one producer attached to it.
        """

        if port is None:
            if self.config.queue_port is None:
                self.config.queue_port = self._get_free_tcp_port(self.host)
                port = self.config.queue_port
            else:
                port = self.config.queue_port

        logger.info(
            f"Adding producer for queue: {queue_name} on: {get_queue_address(port)}"
        )
        producer = ZMQProducer(
            queue_name=queue_name,
            queue_stash=queue_stash,
            port=port,
            context=context,
            worker_stash=worker_stash,
        )
        self.producers[queue_name] = producer
        return producer

    def add_consumer(
        self,
        queue_name: str,
        message_handler: AbstractMessageHandler,
        service_name: str,
        address: str | None = None,
        worker_stash: WorkerStash | None = None,
        syft_worker_id: UID | None = None,
    ) -> ZMQConsumer:
        """Add a consumer to a queue

        A queue should have at least one producer attached to the group.

        """

        if address is None:
            address = get_queue_address(self.config.queue_port)

        consumer = ZMQConsumer(
            queue_name=queue_name,
            message_handler=message_handler,
            address=address,
            service_name=service_name,
            syft_worker_id=syft_worker_id,
            worker_stash=worker_stash,
        )
        self.consumers[queue_name].append(consumer)

        return consumer

    def send_message(
        self,
        message: bytes,
        queue_name: str,
        worker: bytes | None = None,
    ) -> SyftSuccess:
        producer = self.producers.get(queue_name)
        if producer is None:
            raise SyftException(
                public_message=f"No producer attached for queue: {queue_name}. Please add a producer for it."
            )
        try:
            producer.send(message=message, worker=worker)
        except Exception as e:
            # stdlib
            raise SyftException(
                public_message=f"Failed to send message to: {queue_name} with error: {e}"
            )
        return SyftSuccess(
            message=f"Successfully queued message to : {queue_name}",
        )

    def close(self) -> SyftSuccess:
        try:
            for consumers in self.consumers.values():
                for consumer in consumers:
                    # make sure look is stopped
                    consumer.close()

            for producer in self.producers.values():
                # make sure loop is stopped
                producer.close()
                # close existing connection.
        except Exception as e:
            raise SyftException(public_message=f"Failed to close connection: {e}")

        return SyftSuccess(message="All connections closed.")

    def purge_queue(self, queue_name: str) -> SyftSuccess:
        if queue_name not in self.producers:
            raise SyftException(
                public_message=f"No producer running for : {queue_name}"
            )

        producer = self.producers[queue_name]

        # close existing connection.
        producer.close()

        # add a new connection
        self.add_producer(queue_name=queue_name, address=producer.address)  # type: ignore

        return SyftSuccess(message=f"Queue: {queue_name} successfully purged")

    def purge_all(self) -> SyftSuccess:
        for queue_name in self.producers:
            self.purge_queue(queue_name=queue_name)

        return SyftSuccess(message="Successfully purged all queues.")


@serializable(canonical_name="ZMQQueueConfig", version=1)
class ZMQQueueConfig(QueueConfig):
    def __init__(
        self,
        client_type: type[ZMQClient] | None = None,
        client_config: ZMQClientConfig | None = None,
        consumer_type: ConsumerType = ConsumerType.Process,
    ):
        self.client_type = client_type or ZMQClient
        self.client_config: ZMQClientConfig = client_config or ZMQClientConfig()
        self.consumer_type = consumer_type
