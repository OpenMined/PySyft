# stdlib
# stdlib
from binascii import hexlify
from collections import defaultdict
import itertools
import socketserver
import threading
import time
from time import sleep
import traceback
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
from zmq import Frame
from zmq import LINGER
from zmq.error import ContextTerminated
import zmq.green as zmq

# relative
from ...serde.deserialize import _deserialize
from ...serde.serializable import serializable
from ...serde.serialize import _serialize as serialize
from ...service.action.action_object import ActionObject
from ...service.context import AuthedServiceContext
from ...types.syft_migration import migrate
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SYFT_OBJECT_VERSION_3
from ...types.syft_object import SyftObject
from ...types.transforms import drop
from ...types.transforms import make_set_default
from ...types.uid import UID
from ...util.util import get_queue_address
from ..response import SyftError
from ..response import SyftSuccess
from ..worker.worker_pool import ConsumerState
from ..worker.worker_stash import WorkerStash
from .base_queue import AbstractMessageHandler
from .base_queue import QueueClient
from .base_queue import QueueClientConfig
from .base_queue import QueueConfig
from .base_queue import QueueConsumer
from .base_queue import QueueProducer
from .queue_stash import ActionQueueItem
from .queue_stash import Status

HEARTBEAT_LIVENESS = 30  # second
HEARTBEAT_INTERVAL = 2500  # msec
HEARTBEAT_EXPIRY = HEARTBEAT_INTERVAL * HEARTBEAT_LIVENESS
RECONNECT_INTERVAL = 2
INTERVAL_INIT = 1
INTERVAL_MAX = 32
DEFAULT_THREAD_TIMEOUT = 5
POLLER_TIMEOUT = 2500


class QueueMsgProtocol:
    W_WORKER = b"MDPW01"
    W_READY = b"0x01"
    W_REQUEST = b"0x02"
    W_REPLY = b"0x03"
    W_HEARTBEAT = b"0x04"
    W_DISCONNECT = b"0x05"


MAX_RECURSION_NESTED_ACTIONOBJECTS = 5

lock = threading.Lock()


class Worker:
    def __init__(
        self,
        address: str,
        identity: bytes,
        service: Optional[str] = None,
        syft_worker_id: Optional[Union[UID, str]] = None,
    ):
        self.identity = identity
        self.address = address
        self.service = service
        self.expiry = time.time() + 1e-3 * HEARTBEAT_EXPIRY
        self.syft_worker_id = UID(syft_worker_id)


class Service:
    def __init__(self, name: str) -> None:
        self.name = name
        self.requests = []
        self.waiting = []  # List of waiting workers


@serializable()
class ZMQProducer(QueueProducer):
    INTERNAL_SERVICE_PREFIX = b"mmi."

    def __init__(
        self, queue_name: str, queue_stash, port: int, context: AuthedServiceContext
    ) -> None:
        self.id = UID().short()
        self.port = port
        self.queue_stash = queue_stash
        self.queue_name = queue_name
        self.auth_context = context
        self.post_init()
        self._stop = False

    @property
    def address(self):
        return get_queue_address(self.port)

    def post_init(self):
        """Initialize producer state."""

        self.services = {}
        self.workers = {}
        self.waiting: List[Worker] = []
        self.heartbeat_at = time.time() + HEARTBEAT_INTERVAL
        self.context = zmq.Context(1)
        self.backend = self.context.socket(zmq.ROUTER)
        self.backend.setsockopt(LINGER, 1)
        self.backend.setsockopt_string(zmq.IDENTITY, self.id)
        self.poll_workers = zmq.Poller()
        self.poll_workers.register(self.backend, zmq.POLLIN)
        self.bind(f"tcp://*:{self.port}")
        self.thread = None

    def close(self):
        self._stop = True
        try:
            self.poll_workers.unregister(self.backend)
        except Exception as e:
            print("failed to unregister poller", e)
        finally:
            if self.thread is not None:
                self.thread.join(DEFAULT_THREAD_TIMEOUT)
            self.backend.close()
            self.context.destroy()

    @property
    def action_service(self):
        return self.auth_context.node.get_service("ActionService")

    def contains_unresolved_action_objects(self, arg, recursion=0):
        """recursively check collections for unresolved action objects"""
        if isinstance(arg, UID):
            arg = self.action_service.get(self.auth_context, arg).ok()
            return self.contains_unresolved_action_objects(arg, recursion=recursion + 1)
        if isinstance(arg, ActionObject):
            if not arg.syft_resolved:
                res = self.action_service.get(self.auth_context, arg)
                if res.is_err():
                    return True
                arg = res.ok()
                if not arg.syft_resolved:
                    return True
            arg = arg.syft_action_data

        try:
            value = False
            if isinstance(arg, List):
                for elem in arg:
                    value = self.contains_unresolved_action_objects(
                        elem, recursion=recursion + 1
                    )
                    if value:
                        return True
            if isinstance(arg, Dict):
                for elem in arg.values():
                    value = self.contains_unresolved_action_objects(
                        elem, recursion=recursion + 1
                    )
                    if value:
                        return True
            return value
        except Exception as e:
            print(e)
            return True

    def unwrap_nested_actionobjects(self, data):
        """recursively unwraps nested action objects"""

        if isinstance(data, List):
            return [self.unwrap_nested_actionobjects(obj) for obj in data]
        if isinstance(data, Dict):
            return {
                key: self.unwrap_nested_actionobjects(obj) for key, obj in data.items()
            }
        if isinstance(data, ActionObject):
            return data.get()
        return data

    def preprocess_action_arg(self, arg):
        res = self.action_service.get(context=self.auth_context, uid=arg)
        if res.is_err():
            return arg
        action_object = res.ok()
        data = action_object.syft_action_data
        new_data = self.unwrap_nested_actionobjects(data)
        new_action_object = ActionObject.from_obj(new_data, id=action_object.id)
        res = self.action_service.set(
            context=self.auth_context, action_object=new_action_object
        )

    def read_items(self):
        while True:
            if self._stop:
                break
            sleep(1)

            # Items to be queued
            items_to_queue = self.queue_stash.get_by_status(
                self.queue_stash.partition.root_verify_key,
                status=Status.CREATED,
            ).ok()

            items_to_queue = [] if items_to_queue is None else items_to_queue

            # Queue Items that are in the processing state
            items_processing = self.queue_stash.get_by_status(
                self.queue_stash.partition.root_verify_key,
                status=Status.PROCESSING,
            ).ok()

            items_processing = [] if items_processing is None else items_processing

            for item in itertools.chain(items_to_queue, items_processing):
                if item.status == Status.CREATED:
                    if isinstance(item, ActionQueueItem):
                        action = item.kwargs["action"]
                        if self.contains_unresolved_action_objects(
                            action.args
                        ) or self.contains_unresolved_action_objects(action.kwargs):
                            continue
                        for arg in action.args:
                            self.preprocess_action_arg(arg)
                        for _, arg in action.kwargs.items():
                            self.preprocess_action_arg(arg)

                    msg_bytes = serialize(item, to_bytes=True)
                    worker_pool = item.worker_pool.resolve_with_context(
                        self.auth_context
                    )
                    worker_pool = worker_pool.ok()
                    service_name = worker_pool.name
                    service: Service = self.services.get(service_name)

                    # Skip adding message if corresponding service/pool
                    # is not registered.
                    if service is None:
                        continue

                    # append request message to the corresponding service
                    # This list is processed in dispatch method.

                    # TODO: Logic to evaluate the CAN RUN Condition
                    service.requests.append(msg_bytes)
                    item.status = Status.PROCESSING
                    res = self.queue_stash.update(item.syft_client_verify_key, item)
                    if not res.is_ok():
                        print(f"Failed to update queue item: {item}")
                elif item.status == Status.PROCESSING:
                    # Evaluate Retry condition here
                    # If job running and timeout or job status is KILL
                    # or heartbeat fails
                    # or container id doesn't exists, kill process or container
                    # else decrease retry count and mark status as CREATED.
                    pass

    def run(self):
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

        self.producer_thread = threading.Thread(target=self.read_items)
        self.producer_thread.start()

    def send(self, worker: bytes, message: Union[bytes, List[bytes]]):
        worker_obj = self.require_worker(worker)
        self.send_to_worker(worker=worker_obj, msg=message)

    def bind(self, endpoint):
        """Bind producer to endpoint."""
        self.backend.bind(endpoint)
        print("I: MDP producer/0.1.1 is active at %s", endpoint)

    def send_heartbeats(self):
        """Send heartbeats to idle workers if it's time"""
        if time.time() > self.heartbeat_at:
            for worker in self.waiting:
                self.send_to_worker(worker, QueueMsgProtocol.W_HEARTBEAT, None, None)
            self.heartbeat_at = time.time() + HEARTBEAT_INTERVAL

    def purge_workers(self):
        """Look for & kill expired workers.

        Workers are oldest to most recent, so we stop at the first alive worker.
        """
        while len(self.waiting) > 0:
            w = self.waiting[0]
            if w.expiry < time.time():
                self.delete_worker(w, False)
                self.waiting.pop(0)
            else:
                break

    def worker_waiting(self, worker: Worker):
        """This worker is now waiting for work."""
        # Queue to broker and service waiting lists
        self.waiting.append(worker)
        worker.service.waiting.append(worker)
        worker.expiry = time.time() + 1e-3 * HEARTBEAT_EXPIRY
        self.dispatch(worker.service, None)

    def dispatch(self, service: Service, msg: bytes):
        """Dispatch requests to waiting workers as possible"""
        if msg is not None:  # Queue message if any
            service.requests.append(msg)

        self.purge_workers()
        while service.waiting and service.requests:
            # One worker consuming only one message at a time.
            msg = service.requests.pop(0)
            worker = service.waiting.pop(0)
            self.waiting.remove(worker)
            self.send_to_worker(worker, QueueMsgProtocol.W_REQUEST, None, msg)

    def send_to_worker(
        self,
        worker: Worker,
        command: QueueMsgProtocol = QueueMsgProtocol.W_REQUEST,
        option: bytes = None,
        msg: Optional[Union[bytes, list]] = None,
    ):
        """Send message to worker.

        If message is provided, sends that message.
        """

        if msg is None:
            msg = []
        elif not isinstance(msg, list):
            msg = [msg]

        # Stack routing and protocol envelopes to start of message
        # and routing envelope
        if option is not None:
            msg = [option] + msg
        msg = [worker.address, b"", QueueMsgProtocol.W_WORKER, command] + msg

        print("I: sending %r to worker", command)
        with lock:
            self.backend.send_multipart(msg)

    def _run(self):
        while True:
            if self._stop:
                return

            for _, service in self.services.items():
                self.dispatch(service, None)

            items = None

            try:
                items = self.poll_workers.poll(HEARTBEAT_INTERVAL)
            except Exception as e:
                print(f"Failed to poll items. Error: {e}")

            if items:
                msg = self.backend.recv_multipart()

                address = msg.pop(0)
                empty = msg.pop(0)  # noqa: F841
                header = msg.pop(0)

                if header == QueueMsgProtocol.W_WORKER:
                    self.process_worker(address, msg)
                else:
                    print("E: Invalid message.")

            self.purge_workers()
            self.send_heartbeats()

    def require_worker(self, address):
        """Finds the worker (creates if necessary)."""
        identity = hexlify(address)
        worker = self.workers.get(identity)
        if worker is None:
            worker = Worker(identity=identity, address=address)
            self.workers[identity] = worker
            print("I: registering new worker: %s", identity)
        return worker

    def process_worker(self, address: bytes, msg: List[bytes]):
        command = msg.pop(0)

        worker_ready = hexlify(address) in self.workers

        worker = self.require_worker(address)

        if QueueMsgProtocol.W_READY == command:
            service_name = msg.pop(0).decode()
            syft_worker_id = msg.pop(0).decode()
            if worker_ready:
                # Not first command in session or Reserved service name
                self.delete_worker(worker, True)
            else:
                # Attach worker to service and mark as idle
                if service_name not in self.services:
                    service = Service(service_name)
                    self.services[service_name] = service
                else:
                    service = self.services.get(service_name)
                worker.service = service
                worker.syft_worker_id = syft_worker_id
                print(
                    f"New Worker Added: <{worker.identity}> for Service: {service.name}"
                )
                self.worker_waiting(worker)

        elif QueueMsgProtocol.W_HEARTBEAT == command:
            if worker_ready:
                worker.expiry = time.time() + 1e-3 * HEARTBEAT_EXPIRY
            else:
                # extract the syft worker id and worker pool name from the message
                # Get the corresponding worker pool and worker
                # update the status to be unhealthy
                self.delete_worker(worker, True)
        elif QueueMsgProtocol.W_DISCONNECT == command:
            self.delete_worker(worker, False)
        else:
            print("E: Invalid message....")

    def delete_worker(self, worker: Worker, disconnect: bool):
        """Deletes worker from all data structures, and deletes worker."""
        if disconnect:
            self.send_to_worker(worker, QueueMsgProtocol.W_DISCONNECT, None, None)

        if worker.service is not None and worker in worker.service.waiting:
            worker.service.waiting.remove(worker)
        self.workers.pop(worker.identity, None)

    @property
    def alive(self):
        return not self.backend.closed


@serializable(attrs=["_subscriber"])
class ZMQConsumer(QueueConsumer):
    def __init__(
        self,
        message_handler: AbstractMessageHandler,
        address: str,
        queue_name: str,
        service_name: str,
        syft_worker_id: Optional[UID] = None,
        worker_stash: Optional[WorkerStash] = None,
        verbose: bool = False,
    ) -> None:
        self.address = address
        self.message_handler = message_handler
        self.service_name = service_name
        self.queue_name = queue_name
        self.context = zmq.Context()
        self.poller = zmq.Poller()
        self.worker = None
        self.verbose = verbose
        self.id = UID().short()
        self._stop = False
        self.syft_worker_id = syft_worker_id
        self.worker_stash = worker_stash
        self.post_init()

    def reconnect_to_producer(self):
        """Connect or reconnect to producer"""
        if self.worker:
            self.poller.unregister(self.worker)
            self.worker.close()
        self.worker = self.context.socket(zmq.DEALER)
        self.worker.linger = 0
        self.worker.setsockopt_string(zmq.IDENTITY, self.id)
        self.worker.connect(self.address)
        self.poller.register(self.worker, zmq.POLLIN)

        if self.verbose:
            print(f"I: <{self.id}> connecting to broker at {self.address}")

        # Register queue with the producer
        self.send_to_producer(
            QueueMsgProtocol.W_READY,
            self.service_name.encode(),
            [str(self.syft_worker_id).encode()],
        )

        # If liveness hits zero, queue is considered disconnected
        self.liveness = HEARTBEAT_LIVENESS
        self.heartbeat_at = time.time() + HEARTBEAT_INTERVAL

    def post_init(self):
        self.reconnect_to_producer()
        self.thread = None

    def close(self):
        self._stop = True
        try:
            self.poller.unregister(self.worker)
        except Exception as e:
            print("failed to unregister poller", e)
        finally:
            if self.thread is not None:
                self.thread.join(timeout=DEFAULT_THREAD_TIMEOUT)
            self.worker.close()
            self.context.destroy()

    def send_to_producer(
        self,
        command: str,
        option: Optional[bytes] = None,
        msg: Optional[Union[bytes, list]] = None,
    ):
        """Send message to producer.

        If no msg is provided, creates one internally
        """
        if msg is None:
            msg = []
        elif not isinstance(msg, list):
            msg = [msg]

        if option:
            msg = [option] + msg

        msg = [b"", QueueMsgProtocol.W_WORKER, command] + msg
        if self.verbose:
            print("I: sending %s to broker", command)
        with lock:
            self.worker.send_multipart(msg)

    def _run(self):
        """Send reply, if any, to producer and wait for next request."""
        try:
            while True:
                if self._stop:
                    return

                try:
                    items = self.poller.poll(POLLER_TIMEOUT)
                except Exception as e:
                    if isinstance(e, ContextTerminated) or self._stop:
                        return
                    else:
                        print(e, traceback.format_exc())
                        continue

                if items:
                    # Message format:
                    # [b"", "<header>", "<command>", "<queue_name>", "<actual_msg_bytes>"]
                    msg = self.worker.recv_multipart()
                    if self.verbose:
                        print("I: received message from producer: ")
                    self.liveness = HEARTBEAT_LIVENESS

                    if len(msg) < 3:
                        print(f"Invalid message frame. {msg}")
                        continue

                    empty = msg.pop(0)  # noqa: F841
                    header = msg.pop(0)  # noqa: F841

                    command = msg.pop(0)

                    if command == QueueMsgProtocol.W_REQUEST:
                        # Call Message Handler
                        try:
                            message = msg.pop()
                            self.message_handler.handle_message(
                                message=message,
                                syft_worker_id=self.syft_worker_id,
                            )
                        except Exception as e:
                            print(
                                f"ERROR HANDLING MESSAGE: {e}, {traceback.format_exc()}"
                            )
                        finally:
                            self.clear_job()
                    elif command == QueueMsgProtocol.W_HEARTBEAT:
                        pass
                    elif command == QueueMsgProtocol.W_DISCONNECT:
                        self.reconnect_to_producer()
                    else:
                        print("E: invalid input message: ")
                else:
                    self.liveness -= 1
                    if self.liveness == 0:
                        if self.verbose:
                            print("W: disconnected from broker - retrying...")
                        try:
                            time.sleep(RECONNECT_INTERVAL)
                        except Exception as e:
                            print(e, traceback.format_exc())
                            break
                        self.reconnect_to_producer()

                # Send HEARTBEAT if it's time
                if time.time() > self.heartbeat_at:
                    # TODO: Also send service name and syft worker id during HEARTBEATS
                    # to the producer.
                    self.send_to_producer(QueueMsgProtocol.W_HEARTBEAT)
                    self.heartbeat_at = time.time() + HEARTBEAT_INTERVAL

        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                print("Consumer connection Terminated")
            else:
                raise e

        print("W: interrupt received, killing worker...")

    def run(self):
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def associate_job(self, message: Frame):
        try:
            queue_item = _deserialize(message, from_bytes=True)
            self._set_worker_job(queue_item.job_id)
        except Exception as e:
            print("Could not associate job", e)

    def clear_job(self):
        self._set_worker_job(None)

    def _set_worker_job(self, job_id: Optional[UID]):
        if self.worker_stash is not None:
            res = self.worker_stash.get_by_uid(
                self.worker_stash.partition.root_verify_key, self.syft_worker_id
            )

            if res.is_err():
                print(res)
                return  # log/report?

            worker_obj = res.ok()
            worker_obj.job_id = job_id

            if job_id is None:
                worker_obj.consumer_state = ConsumerState.IDLE
            else:
                worker_obj.consumer_state = ConsumerState.CONSUMING

            res = self.worker_stash.update(
                self.worker_stash.partition.root_verify_key, obj=worker_obj
            )
            if res.is_err():
                print(res)

    @property
    def alive(self):
        return not self.worker.closed and self.liveness


@serializable()
class ZMQClientConfigV1(SyftObject, QueueClientConfig):
    __canonical_name__ = "ZMQClientConfig"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    hostname: str = "127.0.0.1"


class ZMQClientConfigV2(SyftObject, QueueClientConfig):
    __canonical_name__ = "ZMQClientConfig"
    __version__ = SYFT_OBJECT_VERSION_2

    id: Optional[UID]
    hostname: str = "127.0.0.1"
    queue_port: Optional[int] = None
    # TODO: setting this to false until we can fix the ZMQ
    # port issue causing tests to randomly fail
    create_producer: bool = False
    n_consumers: int = 0


@serializable()
class ZMQClientConfig(SyftObject, QueueClientConfig):
    __canonical_name__ = "ZMQClientConfig"
    __version__ = SYFT_OBJECT_VERSION_3

    id: Optional[UID]
    hostname: str = "127.0.0.1"
    queue_port: Optional[int] = None
    # TODO: setting this to false until we can fix the ZMQ
    # port issue causing tests to randomly fail
    create_producer: bool = False
    n_consumers: int = 0
    consumer_service: Optional[str]


@migrate(ZMQClientConfig, ZMQClientConfigV1)
def downgrade_zmqclientconfig_v2_to_v1():
    return [
        drop(["queue_port", "create_producer", "n_consumers"]),
    ]


@migrate(ZMQClientConfigV1, ZMQClientConfig)
def upgrade_zmqclientconfig_v1_to_v2():
    return [
        make_set_default("queue_port", None),
        make_set_default("create_producer", False),
        make_set_default("n_consumers", 0),
    ]


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
        self,
        queue_name: str,
        port: Optional[int] = None,
        queue_stash=None,
        context=None,
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
        service_name: str,
        address: Optional[str] = None,
        worker_stash: Optional[WorkerStash] = None,
        syft_worker_id: Optional[UID] = None,
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
        worker: Optional[bytes] = None,
    ) -> Union[SyftSuccess, SyftError]:
        producer = self.producers.get(queue_name)
        if producer is None:
            return SyftError(
                message=f"No producer attached for queue: {queue_name}. Please add a producer for it."
            )
        try:
            producer.send(message=message, worker=worker)
        except Exception as e:
            # stdlib
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
                    # make sure look is stopped
                    consumer.close()

            for _, producer in self.producers.items():
                # make sure loop is stopped
                producer.close()
                # close existing connection.
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
    def __init__(
        self, client_type=None, client_config=None, thread_workers: bool = False
    ):
        self.client_type = client_type or ZMQClient
        self.client_config: ZMQClientConfig = client_config or ZMQClientConfig()
        self.thread_workers = thread_workers
