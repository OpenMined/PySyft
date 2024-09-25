# stdlib
import logging
import subprocess  # nosec
import threading
from threading import Event

# third party
import zmq
from zmq import Frame
from zmq.error import ContextTerminated

# relative
from ...serde.deserialize import _deserialize
from ...serde.serializable import serializable
from ...types.uid import UID
from ..worker.worker_pool import ConsumerState
from ..worker.worker_stash import WorkerStash
from .base_queue import AbstractMessageHandler
from .base_queue import QueueConsumer
from .zmq_common import HEARTBEAT_INTERVAL_SEC
from .zmq_common import PRODUCER_TIMEOUT_SEC
from .zmq_common import THREAD_TIMEOUT_SEC
from .zmq_common import Timeout
from .zmq_common import ZMQCommand
from .zmq_common import ZMQHeader
from .zmq_common import ZMQ_POLLER_TIMEOUT_MSEC
from .zmq_common import ZMQ_SOCKET_LOCK

logger = logging.getLogger(__name__)


def last_created_port() -> int:
    command = (
        "lsof -i -P -n | grep '*:[0-9]* (LISTEN)' | grep python | awk '{print $9, $1, $2}' | "
        "sort -k2,2 -k3,3n | tail -n 1 | awk '{print $1}' | cut -d':' -f2"
    )
    # 1. Lists open files (including network connections) with lsof -i -P -n
    # 2. Filters for listening ports with grep '*:[0-9]* (LISTEN)'
    # 3. Further filters for Python processes with grep python
    # 4. Sorts based on the 9th field (which is likely the port number) with sort -k9
    # 5. Takes the last 10 entries with tail -n 10
    # 6. Prints only the 9th field (port and address) with awk '{print $9}'
    # 7. Extracts only the port number with cut -d':' -f2

    process = subprocess.Popen(  # nosec
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    out, err = process.communicate()
    return int(out.decode("utf-8").strip())


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

    @classmethod
    def default(cls, address: str | None = None, **kwargs: dict) -> "ZMQConsumer":
        # relative
        from ...types.uid import UID
        from ..worker.utils import DEFAULT_WORKER_POOL_NAME
        from .queue import APICallMessageHandler

        if address is None:
            try:
                address = f"tcp://localhost:{last_created_port()}"
            except Exception:
                raise Exception(
                    "Could not auto-assign ZMQConsumer address. Please provide one."
                )
            print(f"Auto-assigning ZMQConsumer address: {address}. Please verify.")
        default_kwargs = {
            "message_handler": APICallMessageHandler,
            "queue_name": APICallMessageHandler.queue_name,
            "service_name": DEFAULT_WORKER_POOL_NAME,
            "syft_worker_id": UID(),
            "verbose": True,
            "address": address,
        }

        for key, value in kwargs.items():
            if key in default_kwargs:
                default_kwargs[key] = value

        return cls(**default_kwargs)

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
            ZMQCommand.W_READY,
            [self.service_name.encode(), str(self.syft_worker_id).encode()],
        )

    def post_init(self) -> None:
        self.thread: threading.Thread | None = None
        self.heartbeat_t = Timeout(HEARTBEAT_INTERVAL_SEC)
        self.producer_ping_t = Timeout(PRODUCER_TIMEOUT_SEC)
        self.reconnect_to_producer()

    def disconnect_from_producer(self) -> None:
        self.send_to_producer(ZMQCommand.W_DISCONNECT)

    def close(self) -> None:
        self.disconnect_from_producer()
        self._stop.set()
        try:
            if self.thread is not None:
                self.thread.join(timeout=THREAD_TIMEOUT_SEC)
                if self.thread is not None and self.thread.is_alive():
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
        core = [b"", ZMQHeader.W_WORKER, command]
        msg = core + msg

        if command != ZMQCommand.W_HEARTBEAT:
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

                    if command != ZMQCommand.W_HEARTBEAT:
                        # log everything except the last frame which contains serialized data
                        logger.info(f"ZMQConsumer recv: {msg[:-4]}")

                    if command == ZMQCommand.W_REQUEST:
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
                    elif command == ZMQCommand.W_HEARTBEAT:
                        self.set_producer_alive()
                    elif command == ZMQCommand.W_DISCONNECT:
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
            self.send_to_producer(ZMQCommand.W_HEARTBEAT)
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
                credentials=self.worker_stash.root_verify_key,
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
