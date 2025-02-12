"""Benchmark class for Syft client."""

from typing import Type

from rich.progress import Progress, SpinnerColumn

from syftbox.client.benchmark import Benchmark, BenchmarkReporter
from syftbox.client.benchmark.network import NetworkBenchmark
from syftbox.client.benchmark.sync import SyncBenchmark
from syftbox.lib.client_config import SyftClientConfig


class SyftBenchmarkRunner:
    """Class to run the benchmark tests for the SyftBox client."""

    def __init__(
        self,
        config: SyftClientConfig,
        reporter: BenchmarkReporter,
    ):
        self.config = config
        self.reporter = reporter

    def get_collectors(self) -> dict[str, Type[Benchmark]]:
        """Get the metric collectors for the benchmark tests."""
        return {
            "network": NetworkBenchmark,
            "sync": SyncBenchmark,
        }

    def run(self, num_runs: int) -> None:
        """Run the benchmark tests."""

        # Get the metric collectors
        collectors = self.get_collectors()

        # Collect all metrics
        metrics = {}
        for name, collector in collectors.items():
            collector_instance = collector(self.config)
            try:
                with Progress(
                    SpinnerColumn(),
                    "{task.description}",
                ) as progress:
                    task = progress.add_task(f"Collecting {name.capitalize()} metrics", total=1)
                    metrics[name] = collector_instance.collect_metrics(num_runs)
                    progress.update(task, completed=True, description=f"Collected {name.capitalize()} metrics")
            except Exception as e:
                print(f"Failed to collect metrics for {name}: {e}")

        # Generate the benchmark report
        self.reporter.generate(metrics)
