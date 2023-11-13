# stdlib
from collections import OrderedDict
from collections import defaultdict
from random import randint
import socketserver
import threading
import time
from typing import DefaultDict
from typing import Dict
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
from .queue_stash import Status
from ...serde.deserialize import _deserialize as deserialize
from ...service.context import AuthedServiceContext
from ...service.job.job_stash import Job
from typing import List
from ..response import SyftNotReady
import functools
from ...service.action.action_object import ActionObject

HEARTBEAT_LIVENESS = 3
HEARTBEAT_INTERVAL = 1
INTERVAL_INIT = 1
INTERVAL_MAX = 32

PPP_READY = b"\x01"  # Signals worker is ready
PPP_HEARTBEAT = b"\x02"  # Signals worker heartbeat

lock = threading.Lock()


class Worker:
    def __init__(self, address):
        self.address = address
        self.expiry = time.time() + HEARTBEAT_INTERVAL * HEARTBEAT_LIVENESS


class WorkerQueue:
    def __init__(self):
        self.queue = OrderedDict()

    def ready(self, worker):
        self.queue.pop(worker.address, None)
        self.queue[worker.address] = worker

    def purge(self):
        """Look for & kill expired workers."""
        t = time.time()
        expired = []
        for address, worker in self.queue.items():
            if t > worker.expiry:  # Worker expired
                expired.append(address)
        for address in expired:
            print("Idle worker expired: %s" % address)
            self.queue.pop(address, None)

    def next(self):
        address, worker = self.queue.popitem(False)
        return address

    def is_empty(self):
        return len(self.queue) == 0


@serializable()
class ZMQProducer(QueueProducer):
    def __init__(self, queue_name: str, queue_stash, port: int, context: AuthedServiceContext) -> None:
        self.port = port
        self.queue_name = queue_name
        self.queue_stash = queue_stash
        self.auth_context = context
        self.post_init()

    @property
    def address(self):
        return f"tcp://localhost:{self.port}"

    def post_init(self):
        self.identity = b"%04X-%04X" % (randint(0, 0x10000), randint(0, 0x10000))
        self.context = zmq.Context(1)
        self.backend = self.context.socket(zmq.ROUTER)  # ROUTER
        self.backend.bind(f"tcp://*:{self.port}")
        self.poll_workers = zmq.Poller()
        self.poll_workers.register(self.backend, zmq.POLLIN)
        self.workers = WorkerQueue()
        self.message_queue = []

    def read_items(self):
        while True:
            # stdlib
            from time import sleep

            sleep(1)
            items = self.queue_stash.get_all(
                self.queue_stash.partition.root_verify_key
            ).ok()
            # syft absolute
            import syft as sy

            for item in items:
                if item.status == Status.CREATED:
                    des = deserialize(item.api_call.serialized_message, from_bytes=True)
                    args = des.args
                    kwargs = des.kwargs
                    import sys
                    
                    action_service = self.auth_context.node.get_service("ActionService")
                    
                    def check_for_job(arg):
                        if isinstance(arg, ActionObject):
                            arg = arg.get()
                        if isinstance(arg, Job):
                            arg.fetch()
                            return arg.resolved
                        try:
                            if isinstance(arg, List):
                                value = True
                                for elem in arg:
                                    value = value and check_for_job(elem)
                                return value
                            if isinstance(arg, Dict):
                                value = True
                                for elem in arg.values():
                                    value = value and check_for_job(elem)
                                return value
                        except Exception as e:
                            pass
                        return True                                           
                        
                    
                    new_args = []
                    for arg in args:
                        new_arg = arg
                        if isinstance(arg, UID):
                            new_arg = action_service.get(self.auth_context, arg).ok()
                        new_args.append(new_arg)
                        
                    print("Kwargs:", kwargs, file=sys.stderr)
                    new_kwargs = {}
                    for kwarg in kwargs:
                        new_kwarg = kwargs[kwarg]
                        if isinstance(kwargs[kwarg], UID):
                            new_kwarg = action_service.get(self.auth_context, kwargs[kwarg]).ok()
                        new_kwargs[kwarg] = new_kwarg
                        if kwarg == 'job_results':
                            # print(job.resolve for job in new_kwarg.get())
                            print("JOBS:", [job.resolve for job in new_kwarg.get()], file=sys.stderr )
                            
                    print("Produder args:", check_for_job(new_args), file=sys.stderr)
                    print("Produder kwargs:", check_for_job(new_kwargs), file=sys.stderr)
                    print(items)
                    
                    if not (check_for_job(new_args) and check_for_job(new_kwargs)):
                        continue                        
                    
                    msg_bytes = sy.serialize(item, to_bytes=True)
                    frames = [self.identity, b"", msg_bytes]
                    # adds to queue for main loop
                    self.message_queue = [frames] + self.message_queue
                    item.status = Status.PROCESSING
                    res = self.queue_stash.update(item.api_call.credentials, item)
                    if not res.is_ok():
                        print("Failed to update queue item")

    def run(self):
        # stdlib

        # self.thread = gevent.spawn(self._run)
        # self.thread.start()

        # self.producer_thread = gevent.spawn(self.read_items)
        # self.producer_thread.start()

        self.thread = threading.Thread(target=self._run)
        self.thread.start()

        self.producer_thread = threading.Thread(target=self.read_items)
        self.producer_thread.start()

    def _run(self):
        heartbeat_at = time.time() + HEARTBEAT_INTERVAL
        connecting_workers = set()
        while True:
            try:
                socks = dict(self.poll_workers.poll(HEARTBEAT_INTERVAL * 1000))

                if len(self.message_queue) != 0:
                    if not self.workers.is_empty():
                        frames = self.message_queue.pop()
                        worker_address = self.workers.next()
                        connecting_workers.add(worker_address)
                        frames.insert(0, worker_address)
                        with lock:
                            self.backend.send_multipart(frames)

                # Handle worker message
                if socks.get(self.backend) == zmq.POLLIN:
                    with lock:
                        frames = self.backend.recv_multipart()
                    if not frames:
                        print("error in producer")
                        break

                    # Validate control message, or return reply to client
                    msg = frames[1:]
                    address = frames[0]
                    if len(msg) == 1:
                        if address not in connecting_workers:
                            self.workers.ready(Worker(address))
                            if msg[0] not in (PPP_READY, PPP_HEARTBEAT):
                                print("E: Invalid message from worker: %s" % msg)
                    else:
                        if address in connecting_workers:
                            connecting_workers.remove(address)
                        # got response message from worker
                        pass

                    # Send heartbeats to idle workers if it's time
                    if time.time() >= heartbeat_at:
                        for worker in self.workers.queue:
                            msg = [worker, PPP_HEARTBEAT]
                            with lock:
                                self.backend.send_multipart(msg)
                        heartbeat_at = time.time() + HEARTBEAT_INTERVAL

                self.workers.purge()
            except Exception as e:
                print(f"Error in producer {e}")

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

    def create_socket(self):
        self.worker = self.ctx.socket(zmq.DEALER)  # DEALER
        self.identity = b"%04X-%04X" % (randint(0, 0x10000), randint(0, 0x10000))
        self.worker.setsockopt(zmq.IDENTITY, self.identity)
        self.poller.register(self.worker, zmq.POLLIN)
        self.worker.connect(self.address)
        self.worker.send(PPP_READY)

    def post_init(self):
        self.ctx = zmq.Context.instance()
        self.poller = zmq.Poller()
        self.create_socket()
        self.thread = None

    def _run(self):
        liveness = HEARTBEAT_LIVENESS
        interval = INTERVAL_INIT
        heartbeat_at = time.time() + HEARTBEAT_INTERVAL
        while True:
            try:
                time.sleep(0.1)
                socks = dict(self.poller.poll(HEARTBEAT_INTERVAL * 1000))
                if socks.get(self.worker) == zmq.POLLIN:
                    with lock:
                        frames = self.worker.recv_multipart()
                    if not frames or len(frames) not in [1, 3]:
                        print(f"Worker error: Invalid message: {frames}")
                        break  # Interrupted

                    # get normal message
                    if len(frames) == 3:
                        with lock:
                            self.worker.send_multipart(frames)
                        liveness = HEARTBEAT_LIVENESS
                        message = frames[2]
                        try:
                            self.message_handler.handle_message(message=message)
                        except Exception as e:
                            # stdlib
                            import traceback

                            print(
                                f"ERROR HANDLING MESSAGE {e}, {traceback.format_exc()}"
                            )
                    # process heartbeat
                    elif len(frames) == 1 and frames[0] == PPP_HEARTBEAT:
                        liveness = HEARTBEAT_LIVENESS
                    # process wrong message
                    interval = INTERVAL_INIT
                # process silence
                else:
                    liveness -= 1
                    if liveness == 0:
                        print(
                            f"Heartbeat failure, worker can't reach queue, reconnecting in {interval}s"
                        )
                        time.sleep(interval)

                        if interval < INTERVAL_MAX:
                            interval *= 2
                        self.poller.unregister(self.worker)
                        self.worker.setsockopt(zmq.LINGER, 0)
                        self.worker.close()
                        self.create_socket()
                        liveness = HEARTBEAT_LIVENESS
                # send heartbeat
                if time.time() > heartbeat_at:
                    heartbeat_at = time.time() + HEARTBEAT_INTERVAL
                    self.worker.send(PPP_HEARTBEAT)
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    print("Subscriber connection Terminated")
                else:
                    raise e

    def run(self):
        # stdlib

        self.thread = threading.Thread(target=self._run)
        self.thread.start()
        # self.thread = gevent.spawn(self._run)
        # self.thread.start()

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
    queue_port: Optional[int] = None
    # TODO: setting this to false until we can fix the ZMQ
    # port issue causing tests to randomly fail
    create_producer: bool = False
    n_consumers: int = 0


@serializable(attrs=["host"])
class ZMQClient(QueueClient):
    """ZMQ Client for creating producers and consumers."""

    producers: Dict[str, ZMQProducer]
    consumers: DefaultDict[str, list[ZMQConsumer]]

    def __init__(self, config: ZMQClientConfig) -> None:
        self.host = config.hostname
        self.producers = {}
        self.consumers = defaultdict(list)
        self.config = config

    @staticmethod
    def _get_free_tcp_port(host: str):
        with socketserver.TCPServer((host, 0), None) as s:
            free_port = s.server_address[1]
        return free_port

    def add_producer(
        self, queue_name: str, port: Optional[int] = None, queue_stash=None, context=None
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

        producer = ZMQProducer(
            queue_name=queue_name, queue_stash=queue_stash, port=port, context=context
        )
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
            address = f"tcp://*:{self.config.queue_port}"

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
    def __init__(self, client_type=None, client_config=None):
        self.client_type = client_type or ZMQClient
        self.client_config: ZMQClientConfig = client_config or ZMQClientConfig()
