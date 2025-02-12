import pytest

from syftbox.client.benchmark import BenchmarkResult
from syftbox.client.benchmark.network import NetworkBenchmark
from syftbox.client.benchmark.runner import SyftBenchmarkRunner
from syftbox.client.benchmark.sync import SyncBenchmark


# Mock classes
class MockResult(BenchmarkResult):
    def __init__(self, num_runs: int):
        self.num_runs = num_runs

    def readable_report(self):
        return "Mock Report"

    def dict_report(self):
        return {"runs": self.num_runs}


@pytest.fixture
def mock_config():
    class MockConfig:
        def __init__(self):
            self.server_url = "https://test.example.com"
            self.access_token = "test-token"
            self.email = "test@example.com"

    return MockConfig()


class MockCollector:
    def __init__(self, config):
        self.config = config

    def collect_metrics(self, num_runs: int):
        return MockResult(num_runs=num_runs)


class MockFailingCollector:
    def __init__(self, config):
        self.config = config

    def collect_metrics(self, num_runs: int):
        raise Exception("Mock collection failure")


class MockReporter:
    def __init__(self):
        self.metrics = None

    def generate(self, metrics):
        self.metrics = metrics


def test_get_collectors():
    """Test getting benchmark collectors"""
    runner = SyftBenchmarkRunner(None, None)
    collectors = runner.get_collectors()

    assert isinstance(collectors, dict)
    assert "network" in collectors
    assert "sync" in collectors
    assert collectors["network"] == NetworkBenchmark
    assert collectors["sync"] == SyncBenchmark


def test_successful_benchmark_run(mock_config, monkeypatch):
    """Test successful benchmark run with all collectors"""
    reporter = MockReporter()
    runner = SyftBenchmarkRunner(mock_config, reporter)

    def mock_get_collectors():
        return {"network": MockCollector, "sync": MockCollector}

    monkeypatch.setattr(runner, "get_collectors", mock_get_collectors)
    runner.run(num_runs=3)

    # Verify metrics were collected and report was generated
    assert reporter.metrics is not None
    assert "network" in reporter.metrics
    assert "sync" in reporter.metrics
    assert isinstance(reporter.metrics["network"], MockResult)
    assert isinstance(reporter.metrics["sync"], MockResult)
    assert reporter.metrics["network"].num_runs == 3
    assert reporter.metrics["sync"].num_runs == 3


def test_partial_benchmark_failure(mock_config, monkeypatch, capsys):
    """Test benchmark run with some failing collectors"""
    reporter = MockReporter()
    runner = SyftBenchmarkRunner(mock_config, reporter)

    def mock_get_collectors():
        return {"network": MockCollector, "sync": MockFailingCollector}

    monkeypatch.setattr(runner, "get_collectors", mock_get_collectors)
    runner.run(num_runs=3)

    # Check error message was printed
    captured = capsys.readouterr()
    assert "Failed to collect metrics for sync" in captured.out

    # Verify successful metrics were collected
    assert reporter.metrics is not None
    assert "network" in reporter.metrics
    assert isinstance(reporter.metrics["network"], MockResult)
    assert reporter.metrics["network"].num_runs == 3


def test_all_benchmarks_failing(mock_config, monkeypatch, capsys):
    """Test benchmark run with all collectors failing"""
    reporter = MockReporter()
    runner = SyftBenchmarkRunner(mock_config, reporter)

    def mock_get_collectors():
        return {"network": MockFailingCollector, "sync": MockFailingCollector}

    monkeypatch.setattr(runner, "get_collectors", mock_get_collectors)
    runner.run(num_runs=3)

    # Check error messages were printed
    captured = capsys.readouterr()
    assert "Failed to collect metrics for network" in captured.out
    assert "Failed to collect metrics for sync" in captured.out

    # Verify empty report was generated
    assert reporter.metrics is not None
    assert isinstance(reporter.metrics, dict)
    assert len(reporter.metrics) == 0


def test_empty_collectors(mock_config, monkeypatch):
    """Test benchmark run with no collectors"""
    reporter = MockReporter()
    runner = SyftBenchmarkRunner(mock_config, reporter)

    def mock_get_collectors():
        return {}

    monkeypatch.setattr(runner, "get_collectors", mock_get_collectors)
    runner.run(num_runs=3)

    # Verify empty report was generated
    assert reporter.metrics is not None
    assert isinstance(reporter.metrics, dict)
    assert len(reporter.metrics) == 0
