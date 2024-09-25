# stdlib
from binascii import hexlify
import itertools
import logging
import sys
import threading
from threading import Event
from time import sleep
from typing import Any

# third party
import zmq
from zmq import LINGER

# relative
from ...serde.serializable import serializable
from ...serde.serialize import _serialize as serialize
from ...service.action.action_object import ActionObject
from ...service.context import AuthedServiceContext
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.uid import UID
from ...util.util import get_queue_address
from ..service import AbstractService
from ..worker.worker_pool import ConsumerState
from ..worker.worker_stash import WorkerStash
from .base_queue import QueueProducer
from .queue_stash import ActionQueueItem
from .queue_stash import QueueStash
from .queue_stash import Status
from .zmq_common import HEARTBEAT_INTERVAL_SEC
from .zmq_common import Service
from .zmq_common import THREAD_TIMEOUT_SEC
from .zmq_common import Timeout
from .zmq_common import Worker
from .zmq_common import ZMQCommand
from .zmq_common import ZMQHeader
from .zmq_common import ZMQ_POLLER_TIMEOUT_MSEC
from .zmq_common import ZMQ_SOCKET_LOCK

logger = logging.getLogger(__name__)


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

    @property
    def action_service(self) -> AbstractService:
        if self.auth_context.server is not None:
            return self.auth_context.server.services.action
        else:
            raise Exception(f"{self.auth_context} does not have a server.")

    @as_result(SyftException)
    def contains_unresolved_action_objects(self, arg: Any, recursion: int = 0) -> bool:
        """recursively check collections for unresolved action objects"""
        if isinstance(arg, UID):
            arg = self.action_service.get(self.auth_context, arg)
            return self.contains_unresolved_action_objects(
                arg, recursion=recursion + 1
            ).unwrap()
        if isinstance(arg, ActionObject):
            if not arg.syft_resolved:
                arg = self.action_service.get(self.auth_context, arg)
                if not arg.syft_resolved:
                    return True
            arg = arg.syft_action_data

        value = False
        if isinstance(arg, list):
            for elem in arg:
                value = self.contains_unresolved_action_objects(
                    elem, recursion=recursion + 1
                ).unwrap()
                if value:
                    return True
        if isinstance(arg, dict):
            for elem in arg.values():
                value = self.contains_unresolved_action_objects(
                    elem, recursion=recursion + 1
                ).unwrap()
                if value:
                    return True
        return value

    def read_items(self) -> None:
        while True:
            if self._stop.is_set():
                break
            try:
                sleep(1)

                # Items to be queued
                items_to_queue = self.queue_stash.get_by_status(
                    self.queue_stash.root_verify_key,
                    status=Status.CREATED,
                ).unwrap()

                items_to_queue = [] if items_to_queue is None else items_to_queue

                # Queue Items that are in the processing state
                items_processing = self.queue_stash.get_by_status(
                    self.queue_stash.root_verify_key,
                    status=Status.PROCESSING,
                ).unwrap()

                items_processing = [] if items_processing is None else items_processing

                for item in itertools.chain(items_to_queue, items_processing):
                    # TODO: if resolving fails, set queueitem to errored, and jobitem as well
                    if item.status == Status.CREATED:
                        if isinstance(item, ActionQueueItem):
                            action = item.kwargs["action"]
                            if (
                                self.contains_unresolved_action_objects(
                                    action.args
                                ).unwrap()
                                or self.contains_unresolved_action_objects(
                                    action.kwargs
                                ).unwrap()
                            ):
                                continue

                        msg_bytes = serialize(item, to_bytes=True)
                        worker_pool = item.worker_pool.resolve_with_context(
                            self.auth_context
                        ).unwrap()
                        service_name = worker_pool.name
                        service: Service | None = self.services.get(service_name)

                        # Skip adding message if corresponding service/pool
                        # is not registered.
                        if service is None:
                            continue

                        # append request message to the corresponding service
                        # This list is processed in dispatch method.

                        # TODO: Logic to evaluate the CAN RUN Condition
                        item.status = Status.PROCESSING
                        self.queue_stash.update(
                            item.syft_client_verify_key, item
                        ).unwrap(public_message=f"failed to update queue item {item}")
                        service.requests.append(msg_bytes)
                    elif item.status == Status.PROCESSING:
                        # Evaluate Retry condition here
                        # If job running and timeout or job status is KILL
                        # or heartbeat fails
                        # or container id doesn't exists, kill process or container
                        # else decrease retry count and mark status as CREATED.
                        pass
            except Exception as e:
                # stdlib
                import traceback

                print(e, traceback.format_exc(), file=sys.stderr)
                item.status = Status.ERRORED
                self.queue_stash.update(item.syft_client_verify_key, item).unwrap()

    def run(self) -> None:
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

        self.producer_thread = threading.Thread(target=self.read_items)
        self.producer_thread.start()

    def send(self, worker: bytes, message: bytes | list[bytes]) -> None:
        worker_obj = self.require_worker(worker)
        self.send_to_worker(worker_obj, ZMQCommand.W_REQUEST, message)

    def bind(self, endpoint: str) -> None:
        """Bind producer to endpoint."""
        self.socket.bind(endpoint)
        logger.info(f"ZMQProducer endpoint: {endpoint}")

    def send_heartbeats(self) -> None:
        """Send heartbeats to idle workers if it's time"""
        if self.heartbeat_t.has_expired():
            for worker in self.waiting:
                self.send_to_worker(worker, ZMQCommand.W_HEARTBEAT)
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

                self.auth_context.server.services.worker._delete(
                    self.auth_context, syft_worker
                )

    def update_consumer_state_for_worker(
        self, syft_worker_id: UID, consumer_state: ConsumerState
    ) -> None:
        if self.worker_stash is None:
            logger.error(  # type: ignore[unreachable]
                f"ZMQProducer worker stash not defined for {self.queue_name} - {self.id}"
            )
            return

        try:
            try:
                self.worker_stash.get_by_uid(
                    credentials=self.worker_stash.root_verify_key,
                    uid=syft_worker_id,
                ).unwrap()
            except Exception:
                return None

            self.worker_stash.update_consumer_state(
                credentials=self.worker_stash.root_verify_key,
                worker_uid=syft_worker_id,
                consumer_state=consumer_state,
            ).unwrap()
        except Exception:
            logger.exception(
                f"Failed to update consumer state for worker id: {syft_worker_id} to state {consumer_state}",
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
            self.send_to_worker(worker, ZMQCommand.W_REQUEST, msg)

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
        core = [worker.address, b"", ZMQHeader.W_WORKER, command]
        msg = core + msg

        if command != ZMQCommand.W_HEARTBEAT:
            # log everything except the last frame which contains serialized data
            logger.info(f"ZMQProducer send: {core}")

        with ZMQ_SOCKET_LOCK:
            try:
                self.socket.send_multipart(msg)
            except zmq.ZMQError:
                logger.exception("ZMQProducer send error")

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

                    if command != ZMQCommand.W_HEARTBEAT:
                        # log everything except the last frame which contains serialized data
                        logger.info(f"ZMQProducer recv: {msg[:4]}")

                    if header == ZMQHeader.W_WORKER:
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

        if ZMQCommand.W_READY == command:
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

        elif ZMQCommand.W_HEARTBEAT == command:
            if worker_ready:
                # If worker is ready then reset expiry
                # and add it to worker waiting list
                # if not already present
                self.worker_waiting(worker)
            else:
                logger.info(f"Got heartbeat, but worker not ready. {worker}")
                self.delete_worker(worker, True)
        elif ZMQCommand.W_DISCONNECT == command:
            logger.info(f"Removing disconnected worker: {worker}")
            self.delete_worker(worker, False)
        else:
            logger.error(f"Invalid command: {command!r}")

    def delete_worker(self, worker: Worker, disconnect: bool) -> None:
        """Deletes worker from all data structures, and deletes worker."""
        if disconnect:
            self.send_to_worker(worker, ZMQCommand.W_DISCONNECT)

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
