import json
from pathlib import Path

from syftbox.client.benchmark import BenchmarkReporter, BenchmarkResult


class JSONReport(BenchmarkReporter):
    """JSON format benchmark report."""

    def __init__(self, path: Path):
        self.output_path = path / "benchmark_report.json"

    def generate(self, metrics: dict[str, BenchmarkResult]) -> None:
        """Generate the benchmark report in JSON format."""

        report: dict = {"result": {}}

        for name, metric in metrics.items():
            report["result"][name] = metric.dict_report()

        with open(self.output_path, "w") as fp:
            json.dump(report, fp, indent=4)

        print("Benchmark result saved at: " + str(self.output_path))


class ConsoleReport(BenchmarkReporter):
    """Human readable format benchmark report"""

    def generate(self, metrics: dict[str, BenchmarkResult]) -> None:
        """Generate the benchmark report in human readable format."""

        report = []
        for name, metric in metrics.items():
            report.append(metric.readable_report())

        print("\n")
        print("\n\n".join(report))
