# stdlib
import binascii
import os
from typing import Any
from typing import Optional

# third party
import gevent
from zmq import Context
from zmq import Socket
import zmq.green as zmq

# relative
from ...serde.deserialize import _deserialize as deserialize
from .queue_stash import QueueItem

PUBLISHER_PORT = 6000
SUBSCRIBER_PORT = 6001


class Publisher:
    def __init__(self, address: str) -> None:
        ctx = zmq.Context.instance()
        self.address = address
        self._publisher = ctx.socket(zmq.PUB)
        self._publisher.bind(address)

    def send(self, message: bytes):
        try:
            # message = [QUEUE_NAME, message_bytes]
            self._publisher.send(message)
            print("Message Send: ", message)
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                print("Connection Interupted....")
            else:
                raise e

    def close(self):
        self._publisher.close()


class Subscriber:
    def __init__(self, worker_settings: Any, address: str, prefix: str = "") -> None:
        ctx = zmq.Context.instance()
        self._subscriber = ctx.socket(zmq.SUB)
        self.address = address
        self._subscriber.connect(address)
        self._subscriber.setsockopt_string(zmq.SUBSCRIBE, prefix)

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

    @staticmethod
    def _receive(subscriber: Socket):
        try:
            return subscriber.recv_multipart()
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                print("Subscriber connection Terminated")
            else:
                raise e

    def _handle_message(self):
        while True:
            message = self._receive(self._subscriber)
            self.process_syft_message(message)

    def run(self, blocking: bool = False):
        # TODO: make this non blocking by running it in a thread
        if blocking:
            return self._handle_message()

        self.recv_thread = gevent.spawn(self._handle_message)
        self.recv_thread.start()

    def process_syft_message(self, message: bytes):
        task_uid, api_call = deserialize(message[0], from_bytes=True, from_proto=False)
        result = self.worker.handle_api_call(api_call)
        item = QueueItem(
            node_uid=self.worker.id, id=task_uid, result=result, resolved=True
        )
        self.worker.queue_stash.set_result(api_call.credentials, item)
        self.worker.queue_stash.partition.close()

    def close(self):
        gevent.sleep(0)
        self.recv_thread.kill()
        self._subscriber.close()


class QueueServer:
    def __init__(self, pub_addr: str, sub_addr: str):
        self.pub_addr = pub_addr
        self.sub_addr = sub_addr
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

    def start(self, in_prefix: bytes = b"pub", out_prefix: bytes = b"sub"):
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

    @staticmethod
    def create(pub_addr: str, sub_addr: str):
        return QueueServer(pub_addr=pub_addr, sub_addr=sub_addr)

    def check_logs(self, timeout: Optional[int]):
        try:
            if self.logger_thread:
                self.logger_thread.join(timeout=timeout)
        except KeyboardInterrupt:
            pass

    def close(self):
        # self.monitored_thread.exit()
        gevent.sleep(0)
        self.thread.kill()
        self.logger_thread.kill()
        self.xpub.close()
        self.xpub.close()
        self.mon_pub.close()
        self.mon_sub.close()
        self.context.destroy()
