# Espresso Pattern

# stdlib
import binascii
import os
from threading import Thread

# third party
import zmq
from zmq.devices import monitored_queue

PUBLISHER_PORT = 6000
SUBSCRIBER_PORT = 6001


class Publisher:
    def __init__(self, port: int, address: str = "tcp://*") -> None:
        ctx = zmq.Context.instance()
        self._publisher = ctx.socket(zmq.PUB)
        self._publisher.bind("tcp://*:6000")

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
    def __init__(self, port: int, address: str = "tcp://localhost") -> None:
        ctx = zmq.Context.instance()
        self._subscriber = ctx.socket(zmq.SUB)
        self._subscriber.connect("tcp://localhost:6001")
        self._subscriber.setsockopt(zmq.SUBSCRIBE, b"")

    def receive(self):
        # TODO: make this non blocking by running it in a thread
        try:
            message = self._subscriber.recv_multipart()
            print("HEllo message recieved: ", message)
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                print("Subscriber connection Terminated")
            else:
                raise e

    def close(self):
        self._subscriber.close()


def zpipe(ctx):
    """build inproc pipe for talking to threads
    mimic pipe used in czmq zthread_fork.
    Returns a pair of PAIRs connected via inproc
    """
    a = ctx.socket(zmq.PAIR)
    b = ctx.socket(zmq.PAIR)
    a.linger = b.linger = 0
    a.hwm = b.hwm = 1
    iface = "inproc://%s" % binascii.hexlify(os.urandom(8))
    a.bind(iface)
    b.connect(iface)
    return a, b


def listener_thread(pipe):
    # Print everything that arrives on pipe
    while True:
        try:
            print(pipe.recv_multipart())
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                break  # Interrupted


def run_mon_queue(queue, *args, **kwargs):
    try:
        queue(*args, **kwargs)
    except Exception as e:
        raise e


class Proxy:
    def __init__(self, address: str, pub_port: int, sub_port: int):
        self.address = address
        self.pub_port = pub_port
        self.sub_port = sub_port
        self.context = zmq.Context.instance()
        self.xsub = self.context.socket(zmq.XSUB)
        self.xpub = self.context.socket(zmq.XPUB)

    def _init_connection(self):
        self.xsub.connect("tcp://localhost:6000")
        self.xpub.bind("tcp://*:6001")
        pipe = zpipe(self.context)
        l_thread = Thread(target=listener_thread, args=(pipe[1],))
        l_thread.start()
        self.monitored_thread = Thread(
            target=run_mon_queue,
            args=(monitored_queue, self.xpub, self.xsub, pipe[0], b"pub", b"sub"),
            daemon=True,
        )
        try:
            self.monitored_thread.start()
        except Exception as e:
            raise e

    def close(self):
        # self.monitored_thread.exit()
        self.xpub.close()
        self.xpub.close()
        self.context.destroy()
