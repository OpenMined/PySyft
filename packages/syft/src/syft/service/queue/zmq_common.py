# stdlib
import threading
import time
from typing import Any

# third party
from pydantic import field_validator

# relative
from ...server.credentials import SyftVerifyKey
from ...types.base import SyftBaseModel
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.uid import UID
from ..worker.worker_pool import SyftWorker
from ..worker.worker_stash import WorkerStash

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

MAX_RECURSION_NESTED_ACTIONOBJECTS = 5


class ZMQHeader:
    """Enum for ZMQ headers"""

    W_WORKER = b"MDPW01"


class ZMQCommand:
    """Enum for ZMQ commands"""

    W_READY = b"0x01"
    W_REQUEST = b"0x02"
    W_REPLY = b"0x03"
    W_HEARTBEAT = b"0x04"
    W_DISCONNECT = b"0x05"


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

    @as_result(SyftException)
    def _syft_worker(
        self, stash: WorkerStash, credentials: SyftVerifyKey
    ) -> SyftWorker | None:
        return stash.get_by_uid(
            credentials=credentials, uid=self.syft_worker_id
        ).unwrap()

    def __str__(self) -> str:
        svc = self.service.name if self.service else None
        return (
            f"Worker(addr={self.address!r}, id={self.identity!r}, service={svc}, "
            f"syft_worker_id={self.syft_worker_id!r})"
        )
