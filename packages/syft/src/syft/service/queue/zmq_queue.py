# stdlib
from collections import defaultdict
import socketserver
from typing import DefaultDict
from typing import Dict
from typing import Optional
from typing import Union

# third party
import gevent
from pydantic import validator
import zmq.green as zmq

# relative
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ..response import SyftError
from ..response import SyftSuccess
from .base_queue import AbstractMessageHandler
from .base_queue import QueueClient
from .base_queue import QueueClientConfig
from .base_queue import QueueConfig
from .base_queue import QueueConsumer
from .base_queue import QueueProducer


@serializable()
class ZMQProducer(QueueProducer):
    def __init__(self, address: str, queue_name: str) -> None:
        ctx = zmq.Context.instance()
        self.address = address
        self._producer = ctx.socket(zmq.PUSH)
        self._producer.bind(address)
        self.queue_name = queue_name

    def send(self, message: bytes) -> None:
        try:
            message_list = [message]
            # TODO: Enable zero copy
            self._producer.send_multipart(message_list)
            print("Message Queued Successfully !")
        except zmq.Again as e:
            # TODO: Add retry mechanism if this error occurs
            raise e
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                print("Connection Interrupted....")
            else:
                raise e

    def close(self):
        self._producer.close()

    @property
    def alive(self):
        return not self._producer.closed


@serializable(attrs=["_subscriber"])
class ZMQConsumer(QueueConsumer):
    def __init__(
        self,
        message_handler: AbstractMessageHandler,
        address: str,
        queue_name: str,
    ) -> None:
        self.address = address
        self.message_handler = message_handler
        self.queue_name = queue_name
        self.post_init()

    def post_init(self):
        ctx = zmq.Context.instance()
        self._consumer = ctx.socket(zmq.PULL)

        self.thread = None
        self._consumer.connect(self.address)

    def receive(self):
        try:
            message_list = self._consumer.recv_multipart()
            message = message_list[0]
            print("Message Received Successfully !")
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                print("Subscriber connection Terminated")
            else:
                raise e
        self.message_handler.handle_message(message=message)

    def _run(self):
        while True:
            self.receive()

    def run(self):
        self.thread = gevent.spawn(self._run)
        self.thread.start()

    def close(self):
        if self.thread is not None:
            self.thread.kill()
        self._consumer.close()

    @property
    def alive(self):
        return not self._consumer.closed


@serializable()
class ZMQClientConfig(SyftObject, QueueClientConfig):
    __canonical_name__ = "ZMQClientConfig"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    hostname: Optional[str]

    @validator("hostname", pre=True, always=True)
    def get_hostname(cls, v: Optional[str]) -> str:
        return "127.0.0.1" if v is None else v


@serializable(attrs=["host"])
class ZMQClient(QueueClient):
    """ZMQ Client for creating producers and consumers."""

    producers: Dict[str, ZMQProducer]
    consumers: DefaultDict[str, list[ZMQConsumer]]

    def __init__(self, config: ZMQClientConfig) -> None:
        self.host = config.hostname
        self.producers = dict()
        self.consumers = defaultdict(list)

    @staticmethod
    def _get_free_tcp_addr(host: str):
        with socketserver.TCPServer((host, 0), None) as s:
            free_port = s.server_address[1]
        addr = f"tcp://{host}:{free_port}"
        return addr

    def add_producer(
        self, queue_name: str, address: Optional[str] = None
    ) -> ZMQProducer:
        """Add a producer of a queue.

        A queue can have at most one producer attached to it.
        """
        if queue_name in self.producers:
            producer = self.producers[queue_name]
            if producer.alive:
                return producer
            address = producer.address
        elif queue_name in self.consumers:
            consumers = self.consumers[queue_name]
            connected_consumers = len(consumers)
            consumer = consumers[0] if connected_consumers > 0 else None
            address = consumer.address if consumer else None

        address = self._get_free_tcp_addr(self.host) if address is None else address
        producer = ZMQProducer(address=address, queue_name=queue_name)
        self.producers[queue_name] = producer

        return producer

    def add_consumer(
        self,
        queue_name: str,
        message_handler: AbstractMessageHandler,
        address: Optional[str] = None,
    ) -> ZMQConsumer:
        """Add a consumer to a queue

        A queue should have at least one producer attached to the group.

        """
        if address is None:
            if queue_name in self.producers:
                address = self.producers[queue_name].address
            elif queue_name in self.consumers:
                consumers = self.consumers[queue_name]
                consumer = consumers[0] if len(consumers) > 0 else None
                address = consumer.address if consumer else None

        address = (
            self._get_free_tcp_addr(
                self.host,
            )
            if address is None
            else address
        )

        consumer = ZMQConsumer(
            queue_name=queue_name,
            message_handler=message_handler,
            address=address,
        )
        self.consumers[queue_name].append(consumer)

        return consumer

    def send_message(
        self,
        message: bytes,
        queue_name: str,
    ) -> Union[SyftSuccess, SyftError]:
        producer = self.producers.get(queue_name)
        if producer is None:
            return SyftError(
                message=f"No producer attached for queue: {queue_name}. Please add a producer for it."
            )
        try:
            producer.send(message=message)
        except Exception as e:
            return SyftError(
                message=f"Failed to send message to: {queue_name} with error: {e}"
            )
        return SyftSuccess(
            message=f"Successfully queued message to : {queue_name}",
        )

    def close(self) -> Union[SyftError, SyftSuccess]:
        try:
            for _, consumers in self.consumers.items():
                for consumer in consumers:
                    consumer.close()

            for _, producer in self.producers.items():
                producer.close()
        except Exception as e:
            return SyftError(message=f"Failed to close connection: {e}")

        return SyftSuccess(message="All connections closed.")

    def purge_queue(self, queue_name: str) -> Union[SyftError, SyftSuccess]:
        if queue_name not in self.producers:
            return SyftError(message=f"No producer running for : {queue_name}")

        producer = self.producers[queue_name]

        # close existing connection.
        producer.close()

        # add a new connection
        self.add_producer(queue_name=queue_name, address=producer.address)

        return SyftSuccess(message=f"Queue: {queue_name} successfully purged")

    def purge_all(self) -> Union[SyftError, SyftSuccess]:
        for queue_name in self.producers:
            self.purge_queue(queue_name=queue_name)

        return SyftSuccess(message="Successfully purged all queues.")


@serializable()
class ZMQQueueConfig(QueueConfig):
    client_config = ZMQClientConfig
    client_type = ZMQClient
