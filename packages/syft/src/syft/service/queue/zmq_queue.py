# stdlib
from collections import defaultdict
import socketserver
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
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
        self._producer = ctx.socket(zmq.REQ)
        self._producer.connect(address)
        self.queue_name = queue_name

    def send(self, message: bytes) -> None:
        try:
            message_list = [message]
            # import ipdb
            # ipdb.set_trace()
            # TODO: Enable zero copy
            self._producer.send_multipart(message_list)
            _ = self._producer.recv()
            # print("Message Queued Successfully !", flush=True)
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
        self.id = UID()

    def post_init(self):
        ctx = zmq.Context.instance()
        self._consumer = ctx.socket(zmq.REP)

        self.thread = None
        self._consumer.connect(self.address)

    def receive(self):
        try:
            print(f"Starting receival ({self.id})")
            message_list = self._consumer.recv_multipart()
            print(f"Received stuff ({self.id})")
            self._consumer.send(b"")
            print(f"sent back confirmation ({self.id})")
            message = message_list[0]
            print("Message Received Successfully !", flush=True)
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
        # stdlib
        import threading

        self.thread = threading.Thread(target=self._run)
        self.thread.start()
        # self.thread = gevent.spawn(self._run)
        # self.thread.start()
        print("spawning thread")

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
    hostname: str = "127.0.0.1"
    consumer_port: Optional[int] = None
    producer_port: Optional[int] = None
    create_message_queue: bool = True


# class MessageQueueConfig():

#     @staticmethod
#     def _get_free_tcp_port(host: str):
#         with socketserver.TCPServer((host, 0), None) as s:
#             free_port = s.server_address[1]
#         return free_port

#     def __init__(self, producer_port: Optional[int]=None,  consumer_port: Optional[int] = None):
#         self.producer_port = producer_port if producer_port is not None else self._get_free_tcp_port(self.host)
#         self.consumer_port = consumer_port


class MessageQueue:
    def __init__(self, consumer_port, producer_port):
        self.consumer_port = consumer_port
        self.producer_port = producer_port
        self.post_init()

    def post_init(self):
        self.thread = None
        self.ctx = zmq.Context.instance()

        # Socket facing clients
        self._frontend = self.ctx.socket(zmq.ROUTER)
        self._frontend.bind(f"tcp://*:{self.producer_port}")

        # Socket facing services
        self._backend = self.ctx.socket(zmq.DEALER)
        self._backend.bind(f"tcp://*:{self.consumer_port}")
        # poller = zmq.Poller()
        # poller.register(frontend, zmq.POLLIN)
        # poller.register(backend, zmq.POLLIN)

    def _run(self):
        zmq.proxy(self._frontend, self._backend)
        # we never get here
        self._frontend.close()
        self._backend.close()
        self.ctx.term()

    def run(self):
        # stdlib
        import threading

        self.thread = threading.Thread(target=self._run)
        self.thread.start()


@serializable(attrs=["host"])
class ZMQClient(QueueClient):
    """ZMQ Client for creating producers and consumers."""

    producers: Dict[str, ZMQProducer]
    consumers: DefaultDict[str, list[ZMQConsumer]]

    def __init__(self, config: ZMQClientConfig) -> None:
        self.host = config.hostname
        self.producers = {}
        self.consumers = defaultdict(list)
        self.message_queue: List[MessageQueue] = None
        self.config = config

    @staticmethod
    def _get_free_tcp_port(host: str):
        with socketserver.TCPServer((host, 0), None) as s:
            free_port = s.server_address[1]
        return free_port

    def add_message_queue(self, queue_name: str):
        if self.config.consumer_port is None:
            self.config.consumer_port = self._get_free_tcp_port(self.host)
        if self.config.producer_port is None:
            self.config.producer_port = self._get_free_tcp_port(self.host)
        self.message_queue = MessageQueue(
            self.config.consumer_port, self.config.producer_port
        )
        return self.message_queue

    def add_producer(
        self, queue_name: str, address: Optional[str] = None
    ) -> ZMQProducer:
        """Add a producer of a queue.

        A queue can have at most one producer attached to it.
        """

        if address is None:
            address = f"tcp://{self.host}:{self.config.producer_port}"

        #     if queue_name in self.producers:
        #         producer = self.producers[queue_name]
        #         if producer.alive:
        #             return producer
        #         address = producer.address

        # if not address:
        #     if self.config.producer_port is None:
        #         self.config.producer_port = port
        #     address = f"tcp://{self.host}:{self.config.producer_port}"

        print(f"CREATING A PRODUCER ON {address}")
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
            address = f"tcp://{self.host}:{self.config.consumer_port}"
        #     if queue_name in self.producers:
        #         address = self.producers[queue_name].address
        #     elif queue_name in self.consumers:
        #         consumers = self.consumers[queue_name]
        #         consumer = consumers[0] if len(consumers) > 0 else None
        #         address = consumer.address if consumer else None

        # address = (
        #     self._get_free_tcp_addr(
        #         self.host,
        #     )
        #     if address is None
        #     else address
        # )

        consumer = ZMQConsumer(
            queue_name=queue_name,
            message_handler=message_handler,
            address=address,
        )
        print(f"CREATING A CONSUMER ON {address}")
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
    client_type = ZMQClient
    client_config: ZMQClientConfig = ZMQClientConfig()
