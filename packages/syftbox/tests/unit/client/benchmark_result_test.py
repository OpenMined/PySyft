from datetime import datetime, timezone

import pytest
import requests

from syftbox.client.benchmark import Stats
from syftbox.client.benchmark.netstats_http import HTTPTimingStats
from syftbox.client.benchmark.netstats_tcp import TCPTimingStats
from syftbox.client.benchmark.network import NetworkBenchmark, NetworkBenchmarkResult
from syftbox.client.benchmark.sync import SyncBenchmark, SyncBenchmarkResult
from syftbox.client.benchmark.syncstats import DataTransferStats


# Mock classes and fixtures
@pytest.fixture
def mock_config():
    class MockConfig:
        def __init__(self):
            self.server_url = "https://test.example.com:8443"
            self.access_token = "test-token"
            self.email = "test@email.com"

    return MockConfig()


@pytest.fixture
def mock_stats():
    return Stats(min=100.0, max=200.0, mean=150.0, stddev=25.0, p50=150.0, p95=190.0, p99=195.0)


@pytest.fixture
def mock_http_stats(mock_stats):
    return HTTPTimingStats(
        dns=mock_stats,
        tcp_connect=mock_stats,
        ssl_handshake=mock_stats,
        send=mock_stats,
        server_wait=mock_stats,
        content=mock_stats,
        total=mock_stats,
        redirect=mock_stats,
        success_rate=95.0,
    )


@pytest.fixture
def mock_tcp_stats(mock_stats):
    return TCPTimingStats(
        latency_stats=mock_stats,
        jitter_stats=mock_stats,
        connection_success_rate=95.0,
        requests_per_minute=30,
        max_requests_per_minute=60,
        max_concurrent_connections=5,
        requests_in_last_minute=25,
    )


# Tests for NetworkBenchmark class
@pytest.mark.parametrize(
    "url,expected_port",
    [
        ("http://example.com", 80),
        ("https://example.com", 443),
        ("http://example.com:8080", 8080),
        ("https://example.com:8443", 8443),
    ],
)
def test_network_benchmark_port_detection(url, expected_port):
    """Test port detection for different URL formats"""

    class CustomConfig:
        def __init__(self, url):
            self.server_url = url

    benchmark = NetworkBenchmark(CustomConfig(url))
    assert benchmark.tcp_perf.port == expected_port


def test_ping_success(mock_config, monkeypatch):
    """Test successful server ping"""

    def mock_get(*args, **kwargs):
        class MockResponse:
            def raise_for_status(self):
                pass

        return MockResponse()

    monkeypatch.setattr(requests, "get", mock_get)

    benchmark = NetworkBenchmark(mock_config)
    assert benchmark.ping() is True


def test_ping_failure(mock_config, monkeypatch):
    """Test failed server ping"""

    def mock_get(*args, **kwargs):
        raise requests.RequestException("Server not reachable")

    monkeypatch.setattr(requests, "get", mock_get)

    benchmark = NetworkBenchmark(mock_config)
    with pytest.raises(requests.RequestException):
        benchmark.ping()


def test_network_collect_metrics(mock_config, mock_http_stats, mock_tcp_stats, monkeypatch):
    """Test metric collection"""

    # Mock ping
    def mock_ping(*args, **kwargs):
        return True

    # Mock HTTP stats
    def mock_http_get_stats(*args, **kwargs):
        return mock_http_stats

    # Mock TCP stats
    def mock_tcp_get_stats(*args, **kwargs):
        return mock_tcp_stats

    benchmark = NetworkBenchmark(mock_config)
    monkeypatch.setattr(benchmark, "ping", mock_ping)
    monkeypatch.setattr(benchmark.http_perf, "get_stats", mock_http_get_stats)
    monkeypatch.setattr(benchmark.tcp_perf, "get_stats", mock_tcp_get_stats)

    result = benchmark.collect_metrics(num_runs=3)

    assert isinstance(result, NetworkBenchmarkResult)
    assert result.url == benchmark.url
    assert result.num_runs == 3
    assert result.http_stats == mock_http_stats
    assert result.tcp_stats == mock_tcp_stats
    assert isinstance(result.timestamp, str)


# Tests for NetworkBenchmarkResult class
def test_network_benchmark_result_formatting(mock_http_stats, mock_tcp_stats):
    """Test benchmark result formatting"""
    result = NetworkBenchmarkResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        num_runs=3,
        url="https://test.example.com",
        http_stats=mock_http_stats,
        tcp_stats=mock_tcp_stats,
    )

    report = result.readable_report()

    # Verify report content
    assert "Network Benchmark" in report
    assert "Server URL" in report
    assert "HTTP Timings" in report
    assert "TCP Timings" in report
    assert str(mock_http_stats.success_rate) in report
    assert str(mock_tcp_stats.connection_success_rate) in report


def test_network_benchmark_error_handling(mock_config, monkeypatch):
    """Test error handling during metric collection"""

    def mock_ping(*args, **kwargs):
        raise requests.RequestException("Server not reachable")

    benchmark = NetworkBenchmark(mock_config)
    monkeypatch.setattr(benchmark, "ping", mock_ping)

    with pytest.raises(requests.RequestException):
        benchmark.collect_metrics(num_runs=3)


def test_network_benchmark_with_empty_stats(mock_config, monkeypatch):
    """Test handling of empty or null statistics"""

    def mock_get_empty_stats(*args, **kwargs):
        raise RuntimeError("No stats available")

    benchmark = NetworkBenchmark(mock_config)
    monkeypatch.setattr(benchmark.http_perf, "get_stats", mock_get_empty_stats)
    monkeypatch.setattr(benchmark.tcp_perf, "get_stats", mock_get_empty_stats)
    monkeypatch.setattr(benchmark, "ping", lambda: True)

    with pytest.raises(RuntimeError):
        benchmark.collect_metrics(num_runs=3)


@pytest.fixture
def mock_data_transfer_stats(mock_stats):
    return DataTransferStats(file_size_mb=1, upload=mock_stats, download=mock_stats, total_runs=3, successful_runs=3)


def test_sync_benchmark_file_sizes():
    """Test benchmark file sizes are properly defined"""
    expected_sizes = [1, 5, 9]  # MB
    assert SyncBenchmark.BENCHMARK_FILE_SIZES == expected_sizes


def test_sync_collect_metrics(mock_config, mock_data_transfer_stats, monkeypatch):
    """Test metric collection for different file sizes"""

    def mock_get_stats(self, size_mb, num_runs):
        return DataTransferStats(
            file_size_mb=size_mb,
            upload=mock_data_transfer_stats.upload,
            download=mock_data_transfer_stats.download,
            total_runs=num_runs,
            successful_runs=num_runs,
        )

    monkeypatch.setattr("syftbox.client.benchmark.syncstats.SyncDataTransferStats.get_stats", mock_get_stats)

    benchmark = SyncBenchmark(mock_config)
    result = benchmark.collect_metrics(num_runs=3)

    assert isinstance(result, SyncBenchmarkResult)
    assert result.url == benchmark.url
    assert result.num_runs == 3
    assert len(result.file_size_stats) == len(benchmark.BENCHMARK_FILE_SIZES)

    # Verify stats for each file size
    for stats, expected_size in zip(result.file_size_stats, benchmark.BENCHMARK_FILE_SIZES):
        assert stats.file_size_mb == expected_size
        assert isinstance(stats.upload, Stats)
        assert isinstance(stats.download, Stats)


def test_sync_benchmark_result_formatting(mock_data_transfer_stats):
    """Test benchmark result formatting"""
    result = SyncBenchmarkResult(url="https://test.example.com", num_runs=3, file_size_stats=[mock_data_transfer_stats])

    report = result.readable_report()

    # Verify report content
    assert "Sync Benchmark" in report
    assert "Server URL" in report
    assert "Runs: 3" in report
    assert "File Size: 1 MB" in report
    assert "Upload Timings" in report
    assert "Download Timings" in report
    assert "Success Rate" in report


def test_collect_metrics_error_handling(mock_config, monkeypatch):
    """Test error handling during metric collection"""

    def mock_get_stats_error(self, size_mb, num_runs):
        raise Exception("Failed to get stats")

    monkeypatch.setattr("syftbox.client.benchmark.syncstats.SyncDataTransferStats.get_stats", mock_get_stats_error)

    benchmark = SyncBenchmark(mock_config)
    with pytest.raises(Exception):
        benchmark.collect_metrics(num_runs=3)


def test_sync_benchmark_result_with_multiple_file_sizes(mock_stats):
    """Test benchmark result with multiple file sizes"""
    file_size_stats = [
        DataTransferStats(
            file_size_mb=size,
            upload=mock_stats,
            download=mock_stats,
            total_runs=3,
            successful_runs=3,
        )
        for size in [1, 5, 9]
    ]

    result = SyncBenchmarkResult(url="https://test.example.com", num_runs=3, file_size_stats=file_size_stats)

    report = result.readable_report()

    # Verify report contains all file sizes
    assert "File Size: 1 MB" in report
    assert "File Size: 5 MB" in report
    assert "File Size: 9 MB" in report


def test_sync_benchmark_with_empty_results(mock_config, monkeypatch):
    """Test handling of empty results"""

    def mock_get_empty_stats(self, size_mb, num_runs):
        return DataTransferStats(
            file_size_mb=size_mb,
            upload=Stats(min=0, max=0, mean=0, stddev=0, p50=0, p95=0, p99=0),
            download=Stats(min=0, max=0, mean=0, stddev=0, p50=0, p95=0, p99=0),
            total_runs=0,
            successful_runs=0,
        )

    monkeypatch.setattr("syftbox.client.benchmark.syncstats.SyncDataTransferStats.get_stats", mock_get_empty_stats)

    benchmark = SyncBenchmark(mock_config)
    result = benchmark.collect_metrics(num_runs=3)

    # Verify empty stats are handled properly
    for stats in result.file_size_stats:
        assert stats.upload.mean == 0
        assert stats.download.mean == 0
        assert stats.total_runs == 0
        assert stats.successful_runs == 0
