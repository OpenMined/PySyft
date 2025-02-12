from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests

from syftbox.client.benchmark import Benchmark, BenchmarkResult
from syftbox.client.benchmark.netstats_http import HTTPPerfStats, HTTPTimingStats
from syftbox.client.benchmark.netstats_tcp import TCPPerfStats, TCPTimingStats
from syftbox.lib.client_config import SyftClientConfig


class NetworkBenchmark(Benchmark):
    """Class for collecting network performance metrics for a server."""

    tcp_perf: TCPPerfStats
    http_perf: HTTPPerfStats

    def __init__(self, config: SyftClientConfig):
        self.url = str(config.server_url)
        parsed = urlparse(self.url)
        host = str(parsed.hostname)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.tcp_perf = TCPPerfStats(host, port)
        self.http_perf = HTTPPerfStats(self.url)

    def collect_metrics(self, num_runs: int) -> "NetworkBenchmarkResult":
        """Calculate network performance metrics."""

        # Check if the server is reachable
        self.ping()

        # Collect HTTP performance stats
        http_stats = self.http_perf.get_stats(num_runs)

        # Collect TCP performance stats
        tcp_stats = self.tcp_perf.get_stats(num_runs)

        return NetworkBenchmarkResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            num_runs=num_runs,
            url=self.url,
            http_stats=http_stats,
            tcp_stats=tcp_stats,
        )

    def ping(self) -> bool:
        """Check if the server is reachable."""
        result = requests.get(str(self.url))
        result.raise_for_status()
        return True


@dataclass
class NetworkBenchmarkResult(BenchmarkResult):
    """Dataclass for network performance metrics."""

    timestamp: str
    url: str
    http_stats: HTTPTimingStats
    tcp_stats: TCPTimingStats

    def readable_report(self) -> str:
        return (
            f"===== Network Benchmark =====\n"
            f"Server URL : {self.url}\n"
            f"Timestamp  : {self.timestamp} UTC\n"
            f"Runs       : {self.num_runs}\n"
            f"\n"
            f"HTTP Timings\n"
            f"DNS (ms)        : {self.http_stats.dns}\n"
            f"Connect (ms)    : {self.http_stats.tcp_connect}\n"
            f"SSL (ms)        : {self.http_stats.ssl_handshake}\n"
            f"Send (ms)       : {self.http_stats.send}\n"
            f"Server (ms)     : {self.http_stats.server_wait}\n"
            f"Download (ms)   : {self.http_stats.content}\n"
            f"Total Time (ms) : {self.http_stats.total}\n"
            f"Redirects (ms)  : {self.http_stats.redirect}\n"
            f"Success Rate    : {self.http_stats.success_rate} %\n"
            "\n"
            f"TCP Timings\n"
            f"Latency (ms) : {self.tcp_stats.latency_stats}\n"
            f"Jitter (ms)  : {self.tcp_stats.jitter_stats}\n"
            f"Success Rate : {self.tcp_stats.connection_success_rate} %"
        )
