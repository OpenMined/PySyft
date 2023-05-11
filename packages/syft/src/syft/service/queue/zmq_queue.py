# stdlib
import binascii
import os
import random
from typing import Any
from typing import Callable
from typing import Optional

# third party
import gevent
from pydantic import validator
from zmq import Context
from zmq import Socket
import zmq.green as zmq

# relative
from ...serde.deserialize import _deserialize as deserialize
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from .base_queue import AbstractMessageHandler
from .base_queue import QueueClient
from .base_queue import QueueClientConfig
from .base_queue import QueueConfig
from .base_queue import QueuePublisher
from .base_queue import QueueSubscriber
from .queue_stash import QueueItem


class ZMQPublisher(QueuePublisher):
    def __init__(self, address: str) -> None:
        ctx = zmq.Context.instance()
        self.address = address
        self._publisher = ctx.socket(zmq.PUB)
        self._publisher.bind(address)

    def send(self, message: bytes, queue_name: str):
        try:
            queue_name_bytes = queue_name.encode()
            message_list = [queue_name_bytes, message]
            self._publisher.send_multipart(message_list)
            print("Message Queued Successfully !")
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                print("Connection Interupted....")
            else:
                raise e

    def close(self):
        self._publisher.close()


class ZMQSubscriber(QueueSubscriber):
    def __init__(
        self,
        worker_settings: Any,
        message_handler: Callable,
        address: str,
        queue_name: str,
    ) -> None:
        ctx = zmq.Context.instance()
        self._subscriber = ctx.socket(zmq.SUB)
        self.address = address
        self.recv_thread = None
        self._subscriber.connect(address)

        self._subscriber.setsockopt_string(zmq.SUBSCRIBE, queue_name)
        self.message_handler = message_handler

        # relative
        from ...node.node import Node

        self.worker = Node(
            id=worker_settings.id,
            name=worker_settings.name,
            signing_key=worker_settings.signing_key,
            document_store_config=worker_settings.document_store_config,
            action_store_config=worker_settings.action_store_config,
            is_subprocess=True,
        )

    def receive(self):
        try:
            message_list = self._subscriber.recv_multipart()
            message = message_list[1]
            print("Message Received Successfully !")
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                print("Subscriber connection Terminated")
            else:
                raise e

        self.message_handler(message=message, worker=self.worker)

    def _run(self):
        while True:
            self.receive()

    def run(self):
        self.recv_thread = gevent.spawn(self._run)
        self.recv_thread.start()

    def close(self):
        if self.recv_thread is not None:
            self.recv_thread.kill()
        self._subscriber.close()


class APICallMessageHandler(AbstractMessageHandler):
    queue = "api_call"

    @classmethod
    def message_handler(cls, message: bytes, worker: Any):
        task_uid, api_call = deserialize(message, from_bytes=True)
        result = worker.handle_api_call(api_call)
        item = QueueItem(
            node_uid=worker.id,
            id=task_uid,
            result=result,
            resolved=True,
        )
        worker.queue_stash.set_result(api_call.credentials, item)
        worker.queue_stash.partition.close()


class ZMQClientConfig(SyftObject, QueueClientConfig):
    __canonical_name__ = "ZMQClientConfig"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    pub_addr: Optional[str]
    sub_addr: Optional[str]

    @staticmethod
    def _get_random_port():
        min_port = 49152
        max_port = 65536
        port = random.randrange(min_port, max_port)
        addr = f"tcp://127.0.0.1:{port}"
        return addr

    @validator("pub_addr", pre=True, always=True)
    def make_pub_addr(cls, v: Optional[str]) -> str:
        return cls._get_random_port() if v is None else v

    @validator("sub_addr", pre=True, always=True)
    def make_sub_addr(cls, v: Optional[str]) -> str:
        return cls._get_random_port() if v is None else v


class ZMQClient(QueueClient):
    def __init__(self, config: QueueClientConfig):
        self.pub_addr = config.pub_addr
        self.sub_addr = config.sub_addr
        self.context = zmq.Context.instance()
        self.logger_thread = None
        self.thread = None

    @staticmethod
    def _setup_monitor(ctx: Context):
        mon_addr = "inproc://%s" % binascii.hexlify(os.urandom(8))
        mon_pub = ctx.socket(zmq.PAIR)
        mon_sub = ctx.socket(zmq.PAIR)

        mon_sub.linger = mon_sub.linger = 0

        mon_sub.hwm = mon_sub.hwm = 1
        mon_pub.bind(mon_addr)
        mon_sub.connect(mon_addr)
        return mon_pub, mon_sub, mon_addr

    def _setup_connections(self):
        self.xsub = self.context.socket(zmq.XSUB)
        self.xpub = self.context.socket(zmq.XPUB)

        self.xsub.connect(self.pub_addr)
        self.xpub.bind(self.sub_addr)

        self.mon_pub, self.mon_sub, self.mon_addr = self._setup_monitor(self.context)

    @staticmethod
    def _start_logger(mon_sub: Socket):
        print("Logging...")
        while True:
            try:
                mon_sub.recv_multipart()
                # message_str = " ".join(mess.decode() for mess in message_bytes)
                # print(message_str)
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    break  # Interrupted

    @staticmethod
    def _start(
        in_socket: Socket,
        out_socket: Socket,
        mon_socket: Socket,
        in_prefix: bytes,
        out_prefix: bytes,
    ):
        poller = zmq.Poller()
        poller.register(in_socket, zmq.POLLIN)
        poller.register(out_socket, zmq.POLLIN)

        while True:
            events = dict(poller.poll())

            if in_socket in events:
                message = in_socket.recv_multipart()
                out_socket.send_multipart(message)
                mon_socket.send_multipart([in_prefix] + message)

            if out_socket in events:
                message = out_socket.recv_multipart()
                in_socket.send_multipart(message)
                mon_socket.send_multipart([out_prefix] + message)

    def start(self, in_prefix: bytes = b"", out_prefix: bytes = b""):
        self._setup_connections()
        self.logger_thread = gevent.spawn(self._start_logger, self.mon_sub)
        self.thread = gevent.spawn(
            self._start,
            self.xpub,
            self.xsub,
            self.mon_pub,
            in_prefix,
            out_prefix,
        )

        self.logger_thread.start()
        self.thread.start()

    def check_logs(self, timeout: Optional[int]):
        try:
            if self.logger_thread:
                self.logger_thread.join(timeout=timeout)
        except KeyboardInterrupt:
            pass

    def close(self):
        self.context.destroy()
        self.xpub.close()
        self.xpub.close()
        self.mon_pub.close()
        self.mon_sub.close()
        self.thread.kill()
        self.logger_thread.kill()


class ZMQQueueConfig(QueueConfig):
    subscriber = ZMQSubscriber
    publisher = ZMQPublisher
    client_config = ZMQClientConfig()
    client_type = ZMQClient
