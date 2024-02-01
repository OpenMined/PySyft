# stdlib
from binascii import hexlify
from collections import defaultdict
import itertools
import socketserver
import threading
import time
from time import sleep
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
from loguru import logger
from pydantic import validator
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
from ...types.base import SyftBaseModel
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

# Producer/Consumer heartbeat interval (in seconds)
HEARTBEAT_INTERVAL_SEC = 2

# Thread join timeout (in seconds)
THREAD_TIMEOUT_SEC = 5

# Max duration (in ms) to wait for ZMQ poller to return
ZMQ_POLLER_TIMEOUT_MSEC = 1000

# Duration (in seconds) after which a worker without a heartbeat will be marked as expired
WORKER_TIMEOUT_SEC = 60

# Duration (in seconds) after which producer without a heartbeat will be marked as expired
PRODUCER_TIMEOUT_SEC = 60

# Lock for working on ZMQ socket
ZMQ_SOCKET_LOCK = threading.Lock()


class QueueMsgProtocol:
    W_WORKER = b"MDPW01"
    W_READY = b"0x01"
    W_REQUEST = b"0x02"
    W_REPLY = b"0x03"
    W_HEARTBEAT = b"0x04"
    W_DISCONNECT = b"0x05"


MAX_RECURSION_NESTED_ACTIONOBJECTS = 5


class Timeout:
    def __init__(self, offset_sec: float):
        self.__offset = float(offset_sec)
        self.__next_ts = 0

        self.reset()

    @property
    def next_ts(self):
        return self.__next_ts

    def reset(self):
        self.__next_ts = self.now() + self.__offset

    def has_expired(self):
        return self.now() >= self.__next_ts

    @staticmethod
    def now() -> float:
        return time.time()


class Worker(SyftBaseModel):
    address: bytes
    identity: bytes
    service: Optional[str] = None
    syft_worker_id: Optional[UID] = None
    expiry_t: Timeout = Timeout(WORKER_TIMEOUT_SEC)

    @validator("syft_worker_id", pre=True, always=True)
    def set_syft_worker_id(cls, v, values):
        if isinstance(v, str):
            return UID(v)
        return v

    def has_expired(self):
        return self.expiry_t.has_expired()

    def get_expiry(self) -> int:
        return self.expiry_t.next_ts

    def reset_expiry(self):
        self.expiry_t.reset()


class Service:
    def __init__(self, name: str) -> None:
        self.name = name
        self.requests = []
        self.waiting = []  # List of waiting workers


@serializable()
class ZMQProducer(QueueProducer):
    INTERNAL_SERVICE_PREFIX = b"mmi."

    def __init__(
        self,
        queue_name: str,
        queue_stash,
        worker_stash: WorkerStash,
        port: int,
        context: AuthedServiceContext,
    ) -> None:
        self.id = UID().short()
        self.port = port
        self.queue_stash = queue_stash
        self.worker_stash = worker_stash
        self.queue_name = queue_name
        self.auth_context = context
        self._stop = threading.Event()
        self.post_init()

    @property
    def address(self):
        return get_queue_address(self.port)

    def post_init(self):
        """Initialize producer state."""

        self.services = {}
        self.workers = {}
        self.waiting: List[Worker] = []
        self.heartbeat_t = Timeout(HEARTBEAT_INTERVAL_SEC)
        self.context = zmq.Context(1)
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.setsockopt(LINGER, 1)
        self.socket.setsockopt_string(zmq.IDENTITY, self.id)
        self.poll_workers = zmq.Poller()
        self.poll_workers.register(self.socket, zmq.POLLIN)
        self.bind(f"tcp://*:{self.port}")
        self.thread: threading.Thread = None
        self.producer_thread: threading.Thread = None

    def close(self):
        self._stop.set()

        try:
            self.poll_workers.unregister(self.socket)
        except Exception as e:
            logger.exception("Failed to unregister poller. {}", e)
        finally:
            if self.thread:
                self.thread.join(THREAD_TIMEOUT_SEC)
                self.thread = None

            if self.producer_thread:
                self.producer_thread.join(THREAD_TIMEOUT_SEC)
                self.producer_thread = None

            self.socket.close()
            self.context.destroy()

            self._stop.clear()

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
            logger.exception("Failed to resolve action objects. {}", e)
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
            res = self.action_service.get(self.auth_context, data.id)
            res = res.ok() if res.is_ok() else res.err()
            if not isinstance(res, ActionObject):
                return SyftError(message=f"{res}")
            else:
                nested_res = res.syft_action_data
                if isinstance(nested_res, ActionObject):
                    nested_res.syft_node_location = res.syft_node_location
                    nested_res.syft_client_verify_key = res.syft_client_verify_key
                return nested_res
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
            if self._stop.is_set():
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
                    if res.is_err():
                        logger.error(
                            "Failed to update queue item={} error={}",
                            item,
                            res.err(),
                        )
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
        self.socket.bind(endpoint)
        logger.info("Producer endpoint: {}", endpoint)

    def send_heartbeats(self):
        """Send heartbeats to idle workers if it's time"""
        if self.heartbeat_t.has_expired():
            for worker in self.waiting:
                self.send_to_worker(worker, QueueMsgProtocol.W_HEARTBEAT, None, None)
            self.heartbeat_t.reset()

    def purge_workers(self):
        """Look for & kill expired workers.

        Workers are oldest to most recent, so we stop at the first alive worker.
        """
        # work on a copy of the iterator
        for worker in list(self.waiting):
            if worker.has_expired():
                logger.info(
                    "Deleting expired Worker id={} uid={} expiry={} now={}",
                    worker.identity,
                    worker.syft_worker_id,
                    worker.get_expiry(),
                    Timeout.now(),
                )
                self.delete_worker(worker, False)

    def update_consumer_state_for_worker(
        self, syft_worker_id: UID, consumer_state: ConsumerState
    ):
        if self.worker_stash is None:
            logger.error(
                f"Worker stash is not defined for ZMQProducer : {self.queue_name} - {self.id}"
            )
            return

        try:
            res = self.worker_stash.update_consumer_state(
                credentials=self.worker_stash.partition.root_verify_key,
                worker_uid=syft_worker_id,
                consumer_state=consumer_state,
            )
            if res.is_err():
                logger.error(
                    "Failed to update consumer state for worker id={} error={}",
                    syft_worker_id,
                    res.err(),
                )
        except Exception as e:
            logger.error(
                f"Failed to update consumer state for worker id: {syft_worker_id}. Error: {e}"
            )

    def worker_waiting(self, worker: Worker):
        """This worker is now waiting for work."""
        # Queue to broker and service waiting lists
        if worker not in self.waiting:
            self.waiting.append(worker)
        if worker not in worker.service.waiting:
            worker.service.waiting.append(worker)
        worker.reset_expiry()
        self.update_consumer_state_for_worker(worker.syft_worker_id, ConsumerState.IDLE)
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

        logger.debug("Send: {}", msg)
        with ZMQ_SOCKET_LOCK:
            self.socket.send_multipart(msg)

    def _run(self):
        while True:
            if self._stop.is_set():
                return

            for _, service in self.services.items():
                self.dispatch(service, None)

            items = None

            try:
                items = self.poll_workers.poll(ZMQ_POLLER_TIMEOUT_MSEC)
            except Exception as e:
                logger.exception("Failed to poll items: {}", e)

            if items:
                msg = self.socket.recv_multipart()

                logger.debug("Recieve: {}", msg)

                address = msg.pop(0)
                empty = msg.pop(0)  # noqa: F841
                header = msg.pop(0)

                if header == QueueMsgProtocol.W_WORKER:
                    self.process_worker(address, msg)
                else:
                    logger.error("Invalid message header: {}", header)

            self.send_heartbeats()
            self.purge_workers()

    def require_worker(self, address):
        """Finds the worker (creates if necessary)."""
        identity = hexlify(address)
        worker = self.workers.get(identity)
        if worker is None:
            worker = Worker(identity=identity, address=address)
            self.workers[identity] = worker
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
                # If worker was already present, then we disconnect it first
                # and wait for it to re-register itself to the producer. This ensures that
                # we always have a healthy worker in place that can talk to the producer.
                self.delete_worker(worker, True)
            else:
                # Attach worker to service and mark as idle
                if service_name not in self.services:
                    service = Service(service_name)
                    self.services[service_name] = service
                else:
                    service = self.services.get(service_name)
                worker.service = service
                worker.syft_worker_id = UID(syft_worker_id)
                logger.info(
                    "New Worker service={} id={} uid={}",
                    service.name,
                    worker.identity,
                    worker.syft_worker_id,
                )
                self.worker_waiting(worker)

        elif QueueMsgProtocol.W_HEARTBEAT == command:
            if worker_ready:
                # If worker is ready then reset expiry
                # and add it to worker waiting list
                # if not already present
                self.worker_waiting(worker)
            else:
                # extract the syft worker id and worker pool name from the message
                # Get the corresponding worker pool and worker
                # update the status to be unhealthy
                self.delete_worker(worker, True)
        elif QueueMsgProtocol.W_DISCONNECT == command:
            self.delete_worker(worker, False)
        else:
            logger.error("Invalid command: {}", command)

    def delete_worker(self, worker: Worker, disconnect: bool):
        """Deletes worker from all data structures, and deletes worker."""
        if disconnect:
            self.send_to_worker(worker, QueueMsgProtocol.W_DISCONNECT, None, None)

        if worker.service and worker in worker.service.waiting:
            worker.service.waiting.remove(worker)

        if worker in self.waiting:
            self.waiting.remove(worker)

        self.workers.pop(worker.identity, None)

        self.update_consumer_state_for_worker(
            worker.syft_worker_id, ConsumerState.DETACHED
        )

    @property
    def alive(self):
        return not self.socket.closed


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
        self.socket = None
        self.verbose = verbose
        self.id = UID().short()
        self._stop = threading.Event()
        self.syft_worker_id = syft_worker_id
        self.worker_stash = worker_stash
        self.post_init()

    def reconnect_to_producer(self):
        """Connect or reconnect to producer"""
        if self.socket:
            self.poller.unregister(self.socket)
            self.socket.close()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.linger = 0
        self.socket.setsockopt_string(zmq.IDENTITY, self.id)
        self.socket.connect(self.address)
        self.poller.register(self.socket, zmq.POLLIN)

        logger.info("Connecting Worker id={} to broker addr={}", self.id, self.address)

        # Register queue with the producer
        self.send_to_producer(
            QueueMsgProtocol.W_READY,
            self.service_name.encode(),
            [str(self.syft_worker_id).encode()],
        )

    def post_init(self):
        self.thread = None
        self.heartbeat_t = Timeout(HEARTBEAT_INTERVAL_SEC)
        self.producer_ping_t = Timeout(PRODUCER_TIMEOUT_SEC)
        self.reconnect_to_producer()

    def close(self):
        self._stop.set()
        try:
            self.poller.unregister(self.socket)
        except Exception as e:
            logger.exception("Failed to unregister worker. {}", e)
        finally:
            if self.thread is not None:
                self.thread.join(timeout=THREAD_TIMEOUT_SEC)
                self.thread = None
            self.socket.close()
            self.context.destroy()
            self._stop.clear()

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
        logger.debug("Send: msg={}", msg)
        with ZMQ_SOCKET_LOCK:
            self.socket.send_multipart(msg)

    def _run(self):
        """Send reply, if any, to producer and wait for next request."""
        try:
            while True:
                if self._stop.is_set():
                    return

                try:
                    items = self.poller.poll(ZMQ_POLLER_TIMEOUT_MSEC)
                except ContextTerminated:
                    logger.info("Context terminated")
                    return
                except Exception as e:
                    logger.error("Poll error={}", e)
                    continue

                if items:
                    # Message format:
                    # [b"", "<header>", "<command>", "<queue_name>", "<actual_msg_bytes>"]
                    msg = self.socket.recv_multipart()

                    logger.debug("Recieve: {}", msg)

                    # mark as alive
                    self.set_producer_alive()

                    if len(msg) < 3:
                        logger.error("Invalid message: {}", msg)
                        continue

                    empty = msg.pop(0)  # noqa: F841
                    header = msg.pop(0)  # noqa: F841

                    command = msg.pop(0)

                    if command == QueueMsgProtocol.W_REQUEST:
                        # Call Message Handler
                        try:
                            message = msg.pop()
                            self.associate_job(message)
                            self.message_handler.handle_message(
                                message=message,
                                syft_worker_id=self.syft_worker_id,
                            )
                        except Exception as e:
                            logger.exception("Error while handling message. {}", e)
                        finally:
                            self.clear_job()
                    elif command == QueueMsgProtocol.W_HEARTBEAT:
                        self.set_producer_alive()
                    elif command == QueueMsgProtocol.W_DISCONNECT:
                        self.reconnect_to_producer()
                    else:
                        logger.error("Invalid command: {}", command)
                else:
                    if not self.is_producer_alive():
                        logger.info("Producer check-alive timed out. Reconnecting.")
                        self.reconnect_to_producer()
                        self.set_producer_alive()

                self.send_heartbeat()

        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                logger.info("Consumer connection terminated")
            else:
                logger.exception("Consumer error. {}", e)
                raise e

        logger.info("Worker finished")

    def set_producer_alive(self):
        self.producer_ping_t.reset()

    def is_producer_alive(self) -> bool:
        # producer timer is within timeout
        return not self.producer_ping_t.has_expired()

    def send_heartbeat(self):
        if self.heartbeat_t.has_expired() and self.is_producer_alive():
            self.send_to_producer(QueueMsgProtocol.W_HEARTBEAT)
            self.heartbeat_t.reset()

    def run(self):
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def associate_job(self, message: Frame):
        try:
            queue_item = _deserialize(message, from_bytes=True)
            self._set_worker_job(queue_item.job_id)
        except Exception as e:
            logger.exception("Could not associate job. {}", e)

    def clear_job(self):
        self._set_worker_job(None)

    def _set_worker_job(self, job_id: Optional[UID]):
        if self.worker_stash is not None:
            consumer_state = (
                ConsumerState.IDLE if job_id is None else ConsumerState.CONSUMING
            )
            res = self.worker_stash.update_consumer_state(
                credentials=self.worker_stash.partition.root_verify_key,
                worker_uid=self.syft_worker_id,
                consumer_state=consumer_state,
            )
            if res.is_err():
                logger.error(
                    f"Failed to update consumer state for {self.service_name}-{self.id}, error={res.err()}"
                )

    @property
    def alive(self):
        return not self.socket.closed and self.is_producer_alive()


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
        worker_stash: Optional[WorkerStash] = None,
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
