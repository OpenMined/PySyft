from dataclasses import dataclass

from syftbox.client.benchmark import Benchmark, BenchmarkResult
from syftbox.client.benchmark.syncstats import DataTransferStats, SyncDataTransferStats
from syftbox.lib.client_config import SyftClientConfig


@dataclass
class SyncBenchmarkResult(BenchmarkResult):
    """Dataclass for sync upload/download performance metrics."""

    url: str
    """URL of the server"""

    file_size_stats: list[DataTransferStats]
    """Data transfer statistics for different file sizes"""

    def readable_report(self) -> str:
        """Generate a human-readable report of the sync benchmark results"""

        report = f"\n===== Sync Benchmark =====\nServer URL : {self.url}\nRuns: {self.num_runs}\n"
        for stats in self.file_size_stats:
            report += (
                f"\n"
                f"File Size: {stats.file_size_mb} MB\n"
                f"Upload Timings (ms): {stats.upload}\n"
                f"Download Timings (ms): {stats.download}\n"
                f"Success Rate: {100 * stats.successful_runs/stats.total_runs} %\n"
            )

        return report


class SyncBenchmark(Benchmark):
    """Class for collecting sync performance metrics for a server"""

    BENCHMARK_FILE_SIZES = [1, 5, 9]  # MB
    sync_perf: SyncDataTransferStats

    def __init__(self, config: SyftClientConfig):
        self.config = config
        self.url = str(config.server_url)
        self.sync_perf = SyncDataTransferStats(
            url=self.url,
            token=str(config.access_token),
            email=config.email,
        )

    def collect_metrics(self, num_runs: int) -> SyncBenchmarkResult:
        """Collect and compile performance metrics for different file sizes"""

        performance_results: list[DataTransferStats] = []

        # Collect performance metrics for different file sizes
        for size_mb in self.BENCHMARK_FILE_SIZES:
            stats = self.sync_perf.get_stats(size_mb, num_runs)
            performance_results.append(stats)

        return SyncBenchmarkResult(
            url=self.url,
            file_size_stats=performance_results,
            num_runs=num_runs,
        )
