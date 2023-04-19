# stdlib
import binascii
import os
from typing import Optional

# third party
import gevent
from zmq import Context
from zmq import Socket
import zmq.green as zmq

# relative
from ...serde.deserialize import _deserialize as deserialize

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
    def __init__(self, address: str, prefix: str = "") -> None:
        ctx = zmq.Context.instance()
        self._subscriber = ctx.socket(zmq.SUB)
        self.address = address
        self._subscriber.connect(address)
        self._subscriber.setsockopt_string(zmq.SUBSCRIBE, prefix)

    @staticmethod
    def _receive(subscriber: Socket):
        try:
            message = subscriber.recv_multipart()
            print("HEllo message received: ", message)
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                print("Subscriber connection Terminated")
            else:
                raise e

    def receive(self, blocking: bool = False):
        # TODO: make this non blocking by running it in a thread
        if blocking:
            return self._receive(self._subscriber)

        self.recv_thread = gevent.spawn(self._receive, self._subscriber)
        self.recv_thread.start()

    def process_syft_msg(message: bytes):
        deserialize(message, from_bytes=True)
        # TODO: Perform the API calls over here.

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
                message_bytes = mon_sub.recv_multipart()
                message_str = " ".join(mess.decode() for mess in message_bytes)
                print(message_str)
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
