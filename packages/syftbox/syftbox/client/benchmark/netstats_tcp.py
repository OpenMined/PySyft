import socket
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta

from typing_extensions import Deque, Generator, Optional

from syftbox.client.benchmark import Stats


@dataclass
class ConnectionMetadata:
    timestamp: datetime
    host: str
    port: int


@dataclass
class TCPTimingStats:
    """TCP performance metrics."""

    latency_stats: Stats
    jitter_stats: Stats
    connection_success_rate: float
    requests_per_minute: int
    max_requests_per_minute: int
    max_concurrent_connections: int
    requests_in_last_minute: int


class RateLimiter:
    """Manages connection rate limiting"""

    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.requests: Deque[datetime] = deque()
        self.lock = threading.Lock()

    def _clean_old_requests(self) -> None:
        """Remove requests older than 1 minute"""
        cutoff = datetime.now() - timedelta(minutes=1)
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

    @contextmanager
    def rate_limit(self) -> Generator[None, None, None]:
        """Context manager for rate limiting"""
        with self.lock:
            self._clean_old_requests()
            while len(self.requests) >= self.max_requests:
                time.sleep(0.1)
                self._clean_old_requests()
            self.requests.append(datetime.now())
            yield


@dataclass
class TCPConnection:
    """Handles single TCP connection measurement"""

    host: str
    port: int
    timeout: float
    previous_latency: Optional[float] = None

    def connect(self) -> tuple[float, float]:
        """Establish TCP connection and measure performance"""
        try:
            start_time = time.time()
            with socket.create_connection((self.host, self.port), timeout=self.timeout):
                latency = (time.time() - start_time) * 1000

                # Calculate jitter
                jitter = 0.0
                if self.previous_latency is not None:
                    jitter = abs(latency - self.previous_latency)

                return latency, jitter

        except socket.error:
            return -1.0, -1.0


class TCPPerfStats:
    """Measure TCP connection performance"""

    max_connections_per_minute: int = 30
    max_concurrent_connections: int = 3
    connection_timeout: float = 10.0
    min_delay_between_requests: float = 0.5

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.__post_init__()

    def __post_init__(self) -> None:
        self.previous_latency: Optional[float] = None
        self.jitter_values: list[float] = []
        self.request_history: Deque[ConnectionMetadata] = deque()
        self.rate_limiter = RateLimiter(self.max_connections_per_minute)
        self.connection_lock = threading.Lock()

    @contextmanager
    def _connection_context(self) -> Generator[None, None, None]:
        """Context manager for connection tracking"""
        metadata = ConnectionMetadata(datetime.now(), self.host, self.port)
        try:
            with self.connection_lock:
                self.request_history.append(metadata)
            yield
        finally:
            # Clean old history
            with self.connection_lock:
                cutoff = datetime.now() - timedelta(minutes=1)
                while self.request_history and self.request_history[0].timestamp < cutoff:
                    self.request_history.popleft()

    def measure_single_connection(self) -> tuple[float, float]:
        """Measure a single TCP connection with rate limiting"""
        with self.rate_limiter.rate_limit():
            with self._connection_context():
                conn = TCPConnection(self.host, self.port, self.connection_timeout, self.previous_latency)
                latency, jitter = conn.connect()

                if latency >= 0:
                    self.previous_latency = latency
                    if jitter >= 0:
                        self.jitter_values.append(jitter)

                time.sleep(self.min_delay_between_requests)
                return latency, jitter

    def get_stats(self, num_runs: int) -> TCPTimingStats:
        """Perform multiple TCP connections and gather statistics."""
        latencies = []
        jitters = []

        # Use ThreadPoolExecutor for parallel connections
        with ThreadPoolExecutor(max_workers=self.max_concurrent_connections) as executor:
            futures = [executor.submit(self.measure_single_connection) for _ in range(num_runs)]

            for future in futures:
                try:
                    latency, jitter = future.result()
                    if latency >= 0:
                        latencies.append(latency)
                    if jitter >= 0:
                        jitters.append(jitter)
                except Exception as e:
                    raise e

        if not latencies:
            raise RuntimeError("No successful TCP measurements")

        return TCPTimingStats(
            latency_stats=self._calculate_stats(latencies),
            jitter_stats=self._calculate_stats(jitters),
            connection_success_rate=len(latencies) / num_runs * 100,
            requests_per_minute=len(self.request_history),
            max_requests_per_minute=self.max_connections_per_minute,
            max_concurrent_connections=self.max_concurrent_connections,
            requests_in_last_minute=len(self.request_history),
        )

    def _calculate_stats(self, values: list[float]) -> Stats:
        if not values:
            return Stats(0, 0, 0, 0, 0, 0, 0)
        return Stats.from_values(values)
