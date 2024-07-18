# stdlib
from binascii import hexlify
from collections import defaultdict
import itertools
import logging
import socketserver
import sys
import threading
from threading import Event
import time
from time import sleep
from typing import Any
from typing import cast

# third party
from pydantic import field_validator
from result import Result
import zmq
from zmq import Frame
from zmq import LINGER
from zmq.error import ContextTerminated

# relative
from ...serde.deserialize import _deserialize
from ...serde.serializable import serializable
from ...serde.serialize import _serialize as serialize
from ...server.credentials import SyftVerifyKey
from ...service.action.action_object import ActionObject
from ...service.context import AuthedServiceContext
from ...types.base import SyftBaseModel
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ...util.util import get_queue_address
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..worker.worker_pool import ConsumerState
from ..worker.worker_pool import SyftWorker
from ..worker.worker_stash import WorkerStash
from .base_queue import AbstractMessageHandler
from .base_queue import QueueClient
from .base_queue import QueueClientConfig
from .base_queue import QueueConfig
from .base_queue import QueueConsumer
from .base_queue import QueueProducer
from .queue_stash import ActionQueueItem
from .queue_stash import QueueStash
from .queue_stash import Status

# Producer/Consumer heartbeat interval (in seconds)
HEARTBEAT_INTERVAL_SEC = 2

# Thread join timeout (in seconds)
THREAD_TIMEOUT_SEC = 30

# Max duration (in ms) to wait for ZMQ poller to return
ZMQ_POLLER_TIMEOUT_MSEC = 1000

# Duration (in seconds) after which a worker without a heartbeat will be marked as expired
WORKER_TIMEOUT_SEC = 60

# Duration (in seconds) after which producer without a heartbeat will be marked as expired
PRODUCER_TIMEOUT_SEC = 60

# Lock for working on ZMQ socket
ZMQ_SOCKET_LOCK = threading.Lock()

logger = logging.getLogger(__name__)


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
        self.__next_ts: float = 0.0

        self.reset()

    @property
    def next_ts(self) -> float:
        return self.__next_ts

    def reset(self) -> None:
        self.__next_ts = self.now() + self.__offset

    def has_expired(self) -> bool:
        return self.now() >= self.__next_ts

    @staticmethod
    def now() -> float:
        return time.time()


class Service:
    def __init__(self, name: str) -> None:
        self.name = name
        self.requests: list[bytes] = []
        self.waiting: list[Worker] = []  # List of waiting workers


class Worker(SyftBaseModel):
    address: bytes
    identity: bytes
    service: Service | None = None
    syft_worker_id: UID | None = None
    expiry_t: Timeout = Timeout(WORKER_TIMEOUT_SEC)

    @field_validator("syft_worker_id", mode="before")
    @classmethod
    def set_syft_worker_id(cls, v: Any) -> Any:
        if isinstance(v, str):
            return UID(v)
        return v

    def has_expired(self) -> bool:
        return self.expiry_t.has_expired()

    def get_expiry(self) -> float:
        return self.expiry_t.next_ts

    def reset_expiry(self) -> None:
        self.expiry_t.reset()

    def _syft_worker(
        self, stash: WorkerStash, credentials: SyftVerifyKey
    ) -> Result[SyftWorker | None, str]:
        return stash.get_by_uid(credentials=credentials, uid=self.syft_worker_id)

    def __str__(self) -> str:
        svc = self.service.name if self.service else None
        return (
            f"Worker(addr={self.address!r}, id={self.identity!r}, service={svc}, "
            f"syft_worker_id={self.syft_worker_id!r})"
        )


@serializable(canonical_name="ZMQProducer", version=1)
class ZMQProducer(QueueProducer):
    INTERNAL_SERVICE_PREFIX = b"mmi."

    def __init__(
        self,
        queue_name: str,
        queue_stash: QueueStash,
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
        self._stop = Event()
        self.post_init()

    @property
    def address(self) -> str:
        return get_queue_address(self.port)

    def post_init(self) -> None:
        """Initialize producer state."""

        self.services: dict[str, Service] = {}
        self.workers: dict[bytes, Worker] = {}
        self.waiting: list[Worker] = []
        self.heartbeat_t = Timeout(HEARTBEAT_INTERVAL_SEC)
        self.context = zmq.Context(1)
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.setsockopt(LINGER, 1)
        self.socket.setsockopt_string(zmq.IDENTITY, self.id)
        self.poll_workers = zmq.Poller()
        self.poll_workers.register(self.socket, zmq.POLLIN)
        self.bind(f"tcp://*:{self.port}")
        self.thread: threading.Thread | None = None
        self.producer_thread: threading.Thread | None = None

    def close(self) -> None:
        self._stop.set()
        try:
            if self.thread:
                self.thread.join(THREAD_TIMEOUT_SEC)
                if self.thread.is_alive():
                    logger.error(
                        f"ZMQProducer message sending thread join timed out during closing. "
                        f"Queue name {self.queue_name}, "
                    )
                self.thread = None

            if self.producer_thread:
                self.producer_thread.join(THREAD_TIMEOUT_SEC)
                if self.producer_thread.is_alive():
                    logger.error(
                        f"ZMQProducer queue thread join timed out during closing. "
                        f"Queue name {self.queue_name}, "
                    )
                self.producer_thread = None

            self.poll_workers.unregister(self.socket)
        except Exception as e:
            logger.exception("Failed to unregister poller.", exc_info=e)
        finally:
            self.socket.close()
            self.context.destroy()

            # self._stop.clear()

    @property
    def action_service(self) -> AbstractService:
        if self.auth_context.server is not None:
            return self.auth_context.server.get_service("ActionService")
        else:
            raise Exception(f"{self.auth_context} does not have a server.")

    def contains_unresolved_action_objects(self, arg: Any, recursion: int = 0) -> bool:
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
            if isinstance(arg, list):
                for elem in arg:
                    value = self.contains_unresolved_action_objects(
                        elem, recursion=recursion + 1
                    )
                    if value:
                        return True
            if isinstance(arg, dict):
                for elem in arg.values():
                    value = self.contains_unresolved_action_objects(
                        elem, recursion=recursion + 1
                    )
                    if value:
                        return True
            return value
        except Exception as e:
            logger.exception("Failed to resolve action objects.", exc_info=e)
            return True

    def read_items(self) -> None:
        while True:
            if self._stop.is_set():
                break
            try:
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
                    # TODO: if resolving fails, set queueitem to errored, and jobitem as well
                    if item.status == Status.CREATED:
                        if isinstance(item, ActionQueueItem):
                            action = item.kwargs["action"]
                            if self.contains_unresolved_action_objects(
                                action.args
                            ) or self.contains_unresolved_action_objects(action.kwargs):
                                continue

                        msg_bytes = serialize(item, to_bytes=True)
                        worker_pool = item.worker_pool.resolve_with_context(
                            self.auth_context
                        )
                        worker_pool = worker_pool.ok()
                        service_name = worker_pool.name
                        service: Service | None = self.services.get(service_name)

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
                                f"Failed to update queue item={item} error={res.err()}"
                            )
                    elif item.status == Status.PROCESSING:
                        # Evaluate Retry condition here
                        # If job running and timeout or job status is KILL
                        # or heartbeat fails
                        # or container id doesn't exists, kill process or container
                        # else decrease retry count and mark status as CREATED.
                        pass
            except Exception as e:
                print(e, file=sys.stderr)
                item.status = Status.ERRORED
                res = self.queue_stash.update(item.syft_client_verify_key, item)
                if res.is_err():
                    logger.error(
                        f"Failed to update queue item={item} error={res.err()}"
                    )

    def run(self) -> None:
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

        self.producer_thread = threading.Thread(target=self.read_items)
        self.producer_thread.start()

    def send(self, worker: bytes, message: bytes | list[bytes]) -> None:
        worker_obj = self.require_worker(worker)
        self.send_to_worker(worker_obj, QueueMsgProtocol.W_REQUEST, message)

    def bind(self, endpoint: str) -> None:
        """Bind producer to endpoint."""
        self.socket.bind(endpoint)
        logger.info(f"ZMQProducer endpoint: {endpoint}")

    def send_heartbeats(self) -> None:
        """Send heartbeats to idle workers if it's time"""
        if self.heartbeat_t.has_expired():
            for worker in self.waiting:
                self.send_to_worker(worker, QueueMsgProtocol.W_HEARTBEAT)
            self.heartbeat_t.reset()

    def purge_workers(self) -> None:
        """Look for & kill expired workers.

        Workers are oldest to most recent, so we stop at the first alive worker.
        """
        # work on a copy of the iterator
        for worker in self.waiting:
            res = worker._syft_worker(self.worker_stash, self.auth_context.credentials)
            if res.is_err() or (syft_worker := res.ok()) is None:
                logger.info(f"Failed to retrieve SyftWorker {worker.syft_worker_id}")
                continue

            if worker.has_expired() or syft_worker.to_be_deleted:
                logger.info(f"Deleting expired worker id={worker}")
                self.delete_worker(worker, syft_worker.to_be_deleted)

                # relative
                from ...service.worker.worker_service import WorkerService

                worker_service = cast(
                    WorkerService, self.auth_context.server.get_service(WorkerService)
                )
                worker_service._delete(self.auth_context, syft_worker)

    def update_consumer_state_for_worker(
        self, syft_worker_id: UID, consumer_state: ConsumerState
    ) -> None:
        if self.worker_stash is None:
            logger.error(  # type: ignore[unreachable]
                f"ZMQProducer worker stash not defined for {self.queue_name} - {self.id}"
            )
            return

        try:
            # Check if worker is present in the database
            worker = self.worker_stash.get_by_uid(
                credentials=self.worker_stash.partition.root_verify_key,
                uid=syft_worker_id,
            )
            if worker.is_ok() and worker.ok() is None:
                return

            res = self.worker_stash.update_consumer_state(
                credentials=self.worker_stash.partition.root_verify_key,
                worker_uid=syft_worker_id,
                consumer_state=consumer_state,
            )
            if res.is_err():
                logger.error(
                    f"Failed to update consumer state for worker id={syft_worker_id} "
                    f"to state: {consumer_state} error={res.err()}",
                )
        except Exception as e:
            logger.error(
                f"Failed to update consumer state for worker id: {syft_worker_id} to state {consumer_state}",
                exc_info=e,
            )

    def worker_waiting(self, worker: Worker) -> None:
        """This worker is now waiting for work."""
        # Queue to broker and service waiting lists
        if worker not in self.waiting:
            self.waiting.append(worker)
        if worker.service is not None and worker not in worker.service.waiting:
            worker.service.waiting.append(worker)
        worker.reset_expiry()
        self.update_consumer_state_for_worker(worker.syft_worker_id, ConsumerState.IDLE)
        self.dispatch(worker.service, None)

    def dispatch(self, service: Service, msg: bytes) -> None:
        """Dispatch requests to waiting workers as possible"""
        if msg is not None:  # Queue message if any
            service.requests.append(msg)

        self.purge_workers()
        while service.waiting and service.requests:
            # One worker consuming only one message at a time.
            msg = service.requests.pop(0)
            worker = service.waiting.pop(0)
            self.waiting.remove(worker)
            self.send_to_worker(worker, QueueMsgProtocol.W_REQUEST, msg)

    def send_to_worker(
        self,
        worker: Worker,
        command: bytes,
        msg: bytes | list | None = None,
    ) -> None:
        """Send message to worker.

        If message is provided, sends that message.
        """

        if self.socket.closed:
            logger.warning("Socket is closed. Cannot send message.")
            return

        if msg is None:
            msg = []
        elif not isinstance(msg, list):
            msg = [msg]

        # ZMQProducer send frames: [address, empty, header, command, ...data]
        core = [worker.address, b"", QueueMsgProtocol.W_WORKER, command]
        msg = core + msg

        if command != QueueMsgProtocol.W_HEARTBEAT:
            # log everything except the last frame which contains serialized data
            logger.info(f"ZMQProducer send: {core}")

        with ZMQ_SOCKET_LOCK:
            try:
                self.socket.send_multipart(msg)
            except zmq.ZMQError as e:
                logger.error("ZMQProducer send error", exc_info=e)

    def _run(self) -> None:
        try:
            while True:
                if self._stop.is_set():
                    logger.info("ZMQProducer thread stopped")
                    return

                for service in self.services.values():
                    self.dispatch(service, None)

                items = None

                try:
                    items = self.poll_workers.poll(ZMQ_POLLER_TIMEOUT_MSEC)
                except Exception as e:
                    logger.exception("ZMQProducer poll error", exc_info=e)

                if items:
                    msg = self.socket.recv_multipart()

                    if len(msg) < 3:
                        logger.error(f"ZMQProducer invalid recv: {msg}")
                        continue

                    # ZMQProducer recv frames: [address, empty, header, command, ...data]
                    (address, _, header, command, *data) = msg

                    if command != QueueMsgProtocol.W_HEARTBEAT:
                        # log everything except the last frame which contains serialized data
                        logger.info(f"ZMQProducer recv: {msg[:4]}")

                    if header == QueueMsgProtocol.W_WORKER:
                        self.process_worker(address, command, data)
                    else:
                        logger.error(f"Invalid message header: {header}")

                self.send_heartbeats()
                self.purge_workers()
        except Exception as e:
            logger.exception("ZMQProducer thread exception", exc_info=e)

    def require_worker(self, address: bytes) -> Worker:
        """Finds the worker (creates if necessary)."""
        identity = hexlify(address)
        worker = self.workers.get(identity)
        if worker is None:
            worker = Worker(identity=identity, address=address)
            self.workers[identity] = worker
        return worker

    def process_worker(self, address: bytes, command: bytes, data: list[bytes]) -> None:
        worker_ready = hexlify(address) in self.workers
        worker = self.require_worker(address)

        if QueueMsgProtocol.W_READY == command:
            service_name = data.pop(0).decode()
            syft_worker_id = data.pop(0).decode()
            if worker_ready:
                # Not first command in session or Reserved service name
                # If worker was already present, then we disconnect it first
                # and wait for it to re-register itself to the producer. This ensures that
                # we always have a healthy worker in place that can talk to the producer.
                self.delete_worker(worker, True)
            else:
                # Attach worker to service and mark as idle
                if service_name in self.services:
                    service: Service | None = self.services.get(service_name)
                else:
                    service = Service(service_name)
                    self.services[service_name] = service
                if service is not None:
                    worker.service = service
                logger.info(f"New worker: {worker}")
                worker.syft_worker_id = UID(syft_worker_id)
                self.worker_waiting(worker)

        elif QueueMsgProtocol.W_HEARTBEAT == command:
            if worker_ready:
                # If worker is ready then reset expiry
                # and add it to worker waiting list
                # if not already present
                self.worker_waiting(worker)
            else:
                logger.info(f"Got heartbeat, but worker not ready. {worker}")
                self.delete_worker(worker, True)
        elif QueueMsgProtocol.W_DISCONNECT == command:
            logger.info(f"Removing disconnected worker: {worker}")
            self.delete_worker(worker, False)
        else:
            logger.error(f"Invalid command: {command!r}")

    def delete_worker(self, worker: Worker, disconnect: bool) -> None:
        """Deletes worker from all data structures, and deletes worker."""
        if disconnect:
            self.send_to_worker(worker, QueueMsgProtocol.W_DISCONNECT)

        if worker.service and worker in worker.service.waiting:
            worker.service.waiting.remove(worker)

        if worker in self.waiting:
            self.waiting.remove(worker)

        self.workers.pop(worker.identity, None)

        if worker.syft_worker_id is not None:
            self.update_consumer_state_for_worker(
                worker.syft_worker_id, ConsumerState.DETACHED
            )

    @property
    def alive(self) -> bool:
        return not self.socket.closed


@serializable(attrs=["_subscriber"], canonical_name="ZMQConsumer", version=1)
class ZMQConsumer(QueueConsumer):
    def __init__(
        self,
        message_handler: AbstractMessageHandler,
        address: str,
        queue_name: str,
        service_name: str,
        syft_worker_id: UID | None = None,
        worker_stash: WorkerStash | None = None,
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
        self._stop = Event()
        self.syft_worker_id = syft_worker_id
        self.worker_stash = worker_stash
        self.post_init()

    def reconnect_to_producer(self) -> None:
        """Connect or reconnect to producer"""
        if self.socket:
            self.poller.unregister(self.socket)  # type: ignore[unreachable]
            self.socket.close()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.linger = 0
        self.socket.setsockopt_string(zmq.IDENTITY, self.id)
        self.socket.connect(self.address)
        self.poller.register(self.socket, zmq.POLLIN)

        logger.info(f"Connecting Worker id={self.id} to broker addr={self.address}")

        # Register queue with the producer
        self.send_to_producer(
            QueueMsgProtocol.W_READY,
            [self.service_name.encode(), str(self.syft_worker_id).encode()],
        )

    def post_init(self) -> None:
        self.thread: threading.Thread | None = None
        self.heartbeat_t = Timeout(HEARTBEAT_INTERVAL_SEC)
        self.producer_ping_t = Timeout(PRODUCER_TIMEOUT_SEC)
        self.reconnect_to_producer()

    def disconnect_from_producer(self) -> None:
        self.send_to_producer(QueueMsgProtocol.W_DISCONNECT)

    def close(self) -> None:
        self.disconnect_from_producer()
        self._stop.set()
        try:
            if self.thread is not None:
                self.thread.join(timeout=THREAD_TIMEOUT_SEC)
                if self.thread.is_alive():
                    logger.error(
                        f"ZMQConsumer thread join timed out during closing. "
                        f"SyftWorker id {self.syft_worker_id}, "
                        f"service name {self.service_name}."
                    )
                self.thread = None
            self.poller.unregister(self.socket)
        except Exception as e:
            logger.error("Failed to unregister worker.", exc_info=e)
        finally:
            self.socket.close()
            self.context.destroy()
            # self._stop.clear()

    def send_to_producer(
        self,
        command: bytes,
        msg: bytes | list | None = None,
    ) -> None:
        """Send message to producer.

        If no msg is provided, creates one internally
        """
        if self.socket.closed:
            logger.warning("Socket is closed. Cannot send message.")
            return

        if msg is None:
            msg = []
        elif not isinstance(msg, list):
            msg = [msg]

        # ZMQConsumer send frames: [empty, header, command, ...data]
        core = [b"", QueueMsgProtocol.W_WORKER, command]
        msg = core + msg

        if command != QueueMsgProtocol.W_HEARTBEAT:
            logger.info(f"ZMQ Consumer send: {core}")

        with ZMQ_SOCKET_LOCK:
            try:
                self.socket.send_multipart(msg)
            except zmq.ZMQError as e:
                logger.error("ZMQConsumer send error", exc_info=e)

    def _run(self) -> None:
        """Send reply, if any, to producer and wait for next request."""
        try:
            while True:
                if self._stop.is_set():
                    logger.info("ZMQConsumer thread stopped")
                    return

                try:
                    items = self.poller.poll(ZMQ_POLLER_TIMEOUT_MSEC)
                except ContextTerminated:
                    logger.info("Context terminated")
                    return
                except Exception as e:
                    logger.error("ZMQ poll error", exc_info=e)
                    continue

                if items:
                    msg = self.socket.recv_multipart()

                    # mark as alive
                    self.set_producer_alive()

                    if len(msg) < 3:
                        logger.error(f"ZMQConsumer invalid recv: {msg}")
                        continue

                    # Message frames recieved by consumer:
                    # [empty, header, command, ...data]
                    (_, _, command, *data) = msg

                    if command != QueueMsgProtocol.W_HEARTBEAT:
                        # log everything except the last frame which contains serialized data
                        logger.info(f"ZMQConsumer recv: {msg[:-4]}")

                    if command == QueueMsgProtocol.W_REQUEST:
                        # Call Message Handler
                        try:
                            message = data.pop()
                            self.associate_job(message)
                            self.message_handler.handle_message(
                                message=message,
                                syft_worker_id=self.syft_worker_id,
                            )
                        except Exception as e:
                            logger.exception("Couldn't handle message", exc_info=e)
                        finally:
                            self.clear_job()
                    elif command == QueueMsgProtocol.W_HEARTBEAT:
                        self.set_producer_alive()
                    elif command == QueueMsgProtocol.W_DISCONNECT:
                        self.reconnect_to_producer()
                    else:
                        logger.error(f"ZMQConsumer invalid command: {command}")
                else:
                    if not self.is_producer_alive():
                        logger.info("Producer check-alive timed out. Reconnecting.")
                        self.reconnect_to_producer()
                        self.set_producer_alive()

                if not self._stop.is_set():
                    self.send_heartbeat()

        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                logger.info("zmq.ETERM")
            else:
                logger.exception("zmq.ZMQError", exc_info=e)
        except Exception as e:
            logger.exception("ZMQConsumer thread exception", exc_info=e)

    def set_producer_alive(self) -> None:
        self.producer_ping_t.reset()

    def is_producer_alive(self) -> bool:
        # producer timer is within timeout
        return not self.producer_ping_t.has_expired()

    def send_heartbeat(self) -> None:
        if self.heartbeat_t.has_expired() and self.is_producer_alive():
            self.send_to_producer(QueueMsgProtocol.W_HEARTBEAT)
            self.heartbeat_t.reset()

    def run(self) -> None:
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def associate_job(self, message: Frame) -> None:
        try:
            queue_item = _deserialize(message, from_bytes=True)
            self._set_worker_job(queue_item.job_id)
        except Exception as e:
            logger.exception("Could not associate job", exc_info=e)

    def clear_job(self) -> None:
        self._set_worker_job(None)

    def _set_worker_job(self, job_id: UID | None) -> None:
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
    def alive(self) -> bool:
        return not self.socket.closed and self.is_producer_alive()


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
    ) -> SyftSuccess | SyftError:
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

    def close(self) -> SyftError | SyftSuccess:
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
            return SyftError(message=f"Failed to close connection: {e}")

        return SyftSuccess(message="All connections closed.")

    def purge_queue(self, queue_name: str) -> SyftError | SyftSuccess:
        if queue_name not in self.producers:
            return SyftError(message=f"No producer running for : {queue_name}")

        producer = self.producers[queue_name]

        # close existing connection.
        producer.close()

        # add a new connection
        self.add_producer(queue_name=queue_name, address=producer.address)  # type: ignore

        return SyftSuccess(message=f"Queue: {queue_name} successfully purged")

    def purge_all(self) -> SyftError | SyftSuccess:
        for queue_name in self.producers:
            self.purge_queue(queue_name=queue_name)

        return SyftSuccess(message="Successfully purged all queues.")


@serializable(canonical_name="ZMQQueueConfig", version=1)
class ZMQQueueConfig(QueueConfig):
    def __init__(
        self,
        client_type: type[ZMQClient] | None = None,
        client_config: ZMQClientConfig | None = None,
        thread_workers: bool = False,
    ):
        self.client_type = client_type or ZMQClient
        self.client_config: ZMQClientConfig = client_config or ZMQClientConfig()
        self.thread_workers = thread_workers
