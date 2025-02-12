import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, quantiles, stdev
from typing import TYPE_CHECKING, Any, Optional

from typing_extensions import Protocol, TypeVar

from syftbox.lib.client_config import SyftClientConfig


@dataclass
class BenchmarkResult:
    """Base class for all metrics with common fields."""

    num_runs: int

    def dict_report(self) -> dict:
        return asdict(self)

    def readable_report(self) -> str:
        raise NotImplementedError


class Benchmark(Protocol):
    """
    Protocol for classes that collect performance metrics.
    """

    client_config: SyftClientConfig

    def __init__(self, config: SyftClientConfig):
        self.client_config = config

    def collect_metrics(self, num_runs: int) -> BenchmarkResult:
        """Calculate performance metrics."""
        ...


class BenchmarkReporter(Protocol):
    """Protocol defining the interface for benchmark result reporters."""

    def generate(self, metrics: dict[str, BenchmarkResult]) -> Any:
        """Generate the benchmark report."""
        ...


@dataclass
class Stats:
    """Common statistics structure."""

    min: float
    max: float
    mean: float
    stddev: float
    p50: float
    p95: float
    p99: float

    @classmethod
    def from_values(cls, values: list) -> "Stats":
        assert len(values) > 1, "At least 2 values are required to calculate"
        values = sorted(values)

        q = quantiles(values, n=100)
        return Stats(
            min=min(values),
            max=max(values),
            mean=mean(values),
            stddev=stdev(values),
            p50=q[49],  # median
            p95=q[94],  # 95th percentile
            p99=q[98],  # 99th percentile
        )

    def as_list(self) -> list:
        return [self.mean, self.stddev, self.min, self.p50, self.p95, self.p99, self.max]

    def __str__(self) -> str:
        return f"{self.mean:.3f} ± {self.stddev:.3f} [min: {self.min:.3f}, p50: {self.p50:.3f}, p95: {self.p95:.3f}, p99: {self.p99:.3f}, max: {self.max:.3f}]"
