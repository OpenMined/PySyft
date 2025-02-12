import socket
import threading
import time
from datetime import datetime, timedelta
from typing import Any

import pytest
from curl_cffi import CurlInfo, CurlOpt

from syftbox.client.benchmark import Stats
from syftbox.client.benchmark.netstats_http import HTTPPerfStats, HTTPTimings, HTTPTimingStats
from syftbox.client.benchmark.netstats_tcp import (
    ConnectionMetadata,
    RateLimiter,
    TCPConnection,
    TCPPerfStats,
    TCPTimingStats,
)


def get_mock_curl_response():
    return {
        CurlInfo.NAMELOOKUP_TIME: 0.1,  # DNS lookup
        CurlInfo.CONNECT_TIME: 0.2,  # Connect
        CurlInfo.APPCONNECT_TIME: 0.3,  # SSL handshake
        CurlInfo.PRETRANSFER_TIME: 0.4,  # Pre-transfer
        CurlInfo.STARTTRANSFER_TIME: 0.5,  # Start transfer
        CurlInfo.TOTAL_TIME: 0.6,  # Total time
        CurlInfo.REDIRECT_TIME: 0.05,  # Redirect time
    }


class MockCurl:
    def __init__(self, should_fail: bool = False):
        self.options = {}
        self.should_fail = should_fail
        self.is_closed = False
        self._mock_response = get_mock_curl_response()

    def setopt(self, option: CurlOpt, value: Any) -> None:
        self.options[str(option)] = value

    def perform(self) -> None:
        if self.should_fail:
            raise Exception("Simulated curl failure")
        if str(CurlOpt.URL) not in self.options:
            raise ValueError("URL not set")
        # Reset closed state when performing new request
        self.is_closed = False

    def getinfo(self, info: CurlInfo) -> float:
        return self._mock_response[info]

    def close(self) -> None:
        self.is_closed = True


def assert_float_equal(a: float, b: float, tolerance: float = 1e-6):
    """Assert that two floats are equal within a tolerance"""
    assert abs(a - b) < tolerance, f"Expected {b}, got {a}"


@pytest.fixture
def mock_successful_curl(monkeypatch):
    mock_instance = MockCurl(should_fail=False)

    def mock_curl_constructor():
        return mock_instance

    monkeypatch.setattr("syftbox.client.benchmark.netstats_http.Curl", mock_curl_constructor)
    return mock_instance


@pytest.fixture
def mock_failing_curl(monkeypatch):
    mock_instance = MockCurl(should_fail=True)

    def mock_curl_constructor():
        return mock_instance

    monkeypatch.setattr("syftbox.client.benchmark.netstats_http.Curl", mock_curl_constructor)
    return mock_instance


def test_curl_cleanup(mock_successful_curl):
    """Test that curl resources are properly cleaned up"""
    stats = HTTPPerfStats("https://example.com")
    stats._HTTPPerfStats__make_request(stats.url)
    assert mock_successful_curl.is_closed is True


def test_http_perf_stats_initialization():
    """Test basic initialization of HTTPPerfStats"""
    url = "https://example.com"
    stats = HTTPPerfStats(url)
    assert stats.url == url
    assert stats.connect_timeout == 30
    assert stats.total_timeout == 60
    assert stats.max_redirects == 5


def test_successful_single_request(mock_successful_curl):
    """Test a single successful HTTP request measurement"""
    stats = HTTPPerfStats("https://example.com")
    timings = stats._HTTPPerfStats__make_request(stats.url)

    assert isinstance(timings, HTTPTimings)
    # Use approximate float comparison for timing values
    assert_float_equal(timings.dns, 100.0)  # 0.1 * 1000
    assert_float_equal(timings.tcp_connect, 100.0)  # (0.2 - 0.1) * 1000
    assert_float_equal(timings.ssl_handshake, 100.0)  # (0.3 - 0.2) * 1000
    assert_float_equal(timings.send, 100.0)  # (0.4 - 0.3) * 1000
    assert_float_equal(timings.server_wait, 100.0)  # (0.5 - 0.4) * 1000
    assert_float_equal(timings.content, 100.0)  # (0.6 - 0.5) * 1000
    assert_float_equal(timings.total, 600.0)  # 0.6 * 1000
    assert_float_equal(timings.redirect, 50.0)  # 0.05 * 1000


def test_failing_request(mock_failing_curl):
    """Test handling of a failed HTTP request"""
    stats = HTTPPerfStats("https://example.com")
    with pytest.raises(Exception, match="Simulated curl failure"):
        stats._HTTPPerfStats__make_request(stats.url)


def test_get_stats_aggregation(mock_successful_curl):
    """Test aggregation of multiple HTTP request measurements"""
    stats = HTTPPerfStats("https://example.com")
    timing_stats = stats.get_stats(n_runs=3)

    assert isinstance(timing_stats, HTTPTimingStats)
    assert timing_stats.success_rate == 100.0

    # Get a fresh timing measurement for comparison
    expected_timing = stats._HTTPPerfStats__make_request(stats.url)

    # Check that all Stats objects are properly calculated
    for field in ["dns", "tcp_connect", "ssl_handshake", "send", "server_wait", "content", "total", "redirect"]:
        stat_obj = getattr(timing_stats, field)
        assert isinstance(stat_obj, Stats)
        # Since all measurements are identical in our mock:
        assert_float_equal(stat_obj.mean, getattr(expected_timing, field))
        assert_float_equal(stat_obj.stddev, 0.0)


def test_no_measurements_error(monkeypatch):
    """Test error handling when no measurements succeed"""

    def mock_make_request(*args, **kwargs):
        return None

    stats = HTTPPerfStats("https://example.com")
    monkeypatch.setattr(stats, "_HTTPPerfStats__make_request", mock_make_request)

    with pytest.raises(RuntimeError, match="No successful measurements"):
        stats.get_stats(n_runs=3)


# Mock classes
class MockSocket:
    def __init__(self, would_fail: bool = False):
        self.would_fail = would_fail
        self.closed = False

    def __enter__(self):
        if self.would_fail:
            raise socket.error("Mock connection failure")
        return self

    def __exit__(self, *args):
        self.closed = True


# Time mocking helpers
class MockTime:
    def __init__(self):
        self.current_time = 0.0
        self.sleep_calls = []

    def time(self):
        self.current_time += 0.1
        return self.current_time

    def sleep(self, seconds):
        self.sleep_calls.append(seconds)


# Fixtures
@pytest.fixture
def mock_time():
    return MockTime()


@pytest.fixture
def mock_successful_socket(monkeypatch):
    def mock_create_connection(*args, **kwargs):
        return MockSocket(would_fail=False)

    monkeypatch.setattr(socket, "create_connection", mock_create_connection)


@pytest.fixture
def mock_failing_socket(monkeypatch):
    def mock_create_connection(*args, **kwargs):
        return MockSocket(would_fail=True)

    monkeypatch.setattr(socket, "create_connection", mock_create_connection)


@pytest.fixture
def setup_time_mocks(monkeypatch, mock_time):
    monkeypatch.setattr(time, "time", mock_time.time)
    monkeypatch.setattr(time, "sleep", mock_time.sleep)
    return mock_time


# Test TCPConnection
def test_successful_connection(mock_successful_socket, setup_time_mocks):
    """Test successful TCP connection measurement"""
    conn = TCPConnection("example.com", 80, timeout=1.0)
    latency, jitter = conn.connect()

    assert abs(latency - 100.0) < 0.001
    assert jitter == 0


def test_failed_connection(mock_failing_socket, setup_time_mocks):
    """Test failed TCP connection handling"""
    conn = TCPConnection("example.com", 80, timeout=1.0)
    latency, jitter = conn.connect()

    assert latency == -1.0
    assert jitter == -1.0


def test_jitter_calculation(mock_successful_socket, setup_time_mocks):
    """Test jitter calculation between consecutive connections"""
    conn = TCPConnection("example.com", 80, timeout=1.0, previous_latency=90.0)
    latency, jitter = conn.connect()

    assert abs(latency - 100.0) < 0.001
    assert abs(jitter - 10.0) < 0.001


# Test TCPPerfStats
def test_tcp_perf_stats_initialization():
    """Test TCPPerfStats initialization"""
    stats = TCPPerfStats("example.com", 80)

    assert stats.host == "example.com"
    assert stats.port == 80
    assert stats.previous_latency is None
    assert isinstance(stats.rate_limiter, RateLimiter)
    assert isinstance(stats.connection_lock, type(threading.Lock()))


@pytest.mark.parametrize("num_runs", [3, 0])
def test_get_stats_measurement(mock_successful_socket, setup_time_mocks, num_runs):
    """Test statistics gathering with different numbers of runs"""
    stats = TCPPerfStats("example.com", 80)

    if num_runs > 0:
        result = stats.get_stats(num_runs)
        assert isinstance(result, TCPTimingStats)
        assert isinstance(result.latency_stats, Stats)
        assert isinstance(result.jitter_stats, Stats)
        assert result.connection_success_rate == 100.0
        assert result.requests_per_minute == num_runs
        assert result.max_requests_per_minute == stats.max_connections_per_minute
        assert result.max_concurrent_connections == stats.max_concurrent_connections
        assert result.requests_in_last_minute == num_runs
    else:
        with pytest.raises(RuntimeError, match="No successful TCP measurements"):
            stats.get_stats(num_runs)


def test_concurrent_connections(mock_successful_socket, setup_time_mocks, monkeypatch):
    """Test concurrent connection handling"""

    def mock_thread_pool(*args, **kwargs):
        class MockPool:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def submit(self, fn, *args):
                class MockFuture:
                    def result(self):
                        return fn()

                return MockFuture()

        return MockPool()

    monkeypatch.setattr("concurrent.futures.ThreadPoolExecutor", mock_thread_pool)

    stats = TCPPerfStats("example.com", 80)
    stats.max_concurrent_connections = 2
    result = stats.get_stats(4)

    assert isinstance(result, TCPTimingStats)
    assert result.max_concurrent_connections == 2
    assert result.requests_per_minute == 4
    assert result.requests_in_last_minute == 4
    assert isinstance(result.latency_stats, Stats)
    assert isinstance(result.jitter_stats, Stats)


def test_connection_history_cleaning(setup_time_mocks):
    """Test cleaning of old connection history"""
    stats = TCPPerfStats("example.com", 80)

    old_time = datetime.now() - timedelta(minutes=2)
    stats.request_history.append(ConnectionMetadata(old_time, "example.com", 80))
    stats.request_history.append(ConnectionMetadata(datetime.now(), "example.com", 80))

    with stats._connection_context():
        pass

    assert len(stats.request_history) == 2
    assert all(conn.timestamp > datetime.now() - timedelta(minutes=1) for conn in stats.request_history)


def test_calculate_stats_empty():
    """Test stats calculation with empty values"""
    stats = TCPPerfStats("example.com", 80)
    result = stats._calculate_stats([])

    assert isinstance(result, Stats)
    assert result.mean == 0
    # Only check for attributes that exist in the Stats class
    assert hasattr(result, "min")
    assert hasattr(result, "max")
    assert hasattr(result, "mean")


def test_calculate_stats_values():
    """Test stats calculation with actual values"""
    stats = TCPPerfStats("example.com", 80)
    values = [100.0, 110.0, 90.0]
    result = stats._calculate_stats(values)

    assert isinstance(result, Stats)
    assert result.mean == 100.0
    assert result.min == 90.0
    assert result.max == 110.0


def test_failed_measurements_handling(mock_failing_socket, setup_time_mocks):
    """Test handling of completely failed measurements"""
    stats = TCPPerfStats("example.com", 80)
    with pytest.raises(RuntimeError, match="No successful TCP measurements"):
        stats.get_stats(3)


def test_get_stats_with_mixed_success(monkeypatch, setup_time_mocks):
    """Test statistics gathering with both successful and failed connections"""

    # Create a stateful mock for TCPConnection that alternates between success and failure
    class MockConnection:
        def __init__(self):
            self.call_count = 0

        def connect(self):
            self.call_count += 1
            if self.call_count % 2 == 0:
                return (-1.0, -1.0)  # Simulate failure
            return (100.0, 0.0)  # Successful measurement

    mock_conn = MockConnection()

    def mock_tcp_connection(*args, **kwargs):
        return mock_conn

    monkeypatch.setattr("syftbox.client.benchmark.netstats_tcp.TCPConnection", mock_tcp_connection)

    stats = TCPPerfStats("example.com", 80)
    result = stats.get_stats(4)

    assert isinstance(result, TCPTimingStats)
    assert result.connection_success_rate == 50.0  # Half of the connections succeeded
    assert result.requests_per_minute == 4  # Total attempts
    assert result.requests_in_last_minute == 4
    assert isinstance(result.latency_stats, Stats)
    assert isinstance(result.jitter_stats, Stats)
