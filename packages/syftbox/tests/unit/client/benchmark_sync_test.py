import time

import pytest
import requests

from syftbox.client.benchmark import Stats
from syftbox.client.benchmark.syncstats import (
    DataTransferStats,
    FileTransferDuration,
    SyncDataTransferStats,
    generate_byte_string,
    random_filename,
)


# Mock classes
class MockResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.RequestException(f"HTTP Error: {self.status_code}")


# Fixtures
@pytest.fixture
def sync_stats():
    return SyncDataTransferStats(url="https://example.com", token="test-token", email="test@example.com")


@pytest.fixture
def mock_time(monkeypatch):
    class MockTime:
        def __init__(self):
            self.current_time = 0.0
            self.sleep_calls = []

        def time(self):
            self.current_time += 0.5
            return self.current_time

        def sleep(self, seconds):
            self.sleep_calls.append(seconds)

    mock_timer = MockTime()
    monkeypatch.setattr(time, "time", mock_timer.time)
    monkeypatch.setattr(time, "sleep", mock_timer.sleep)
    return mock_timer


def test_generate_byte_string():
    """Test byte string generation"""
    size_mb = 2
    data = generate_byte_string(size_mb)
    expected_size = size_mb * 1024 * 1024

    assert isinstance(data, bytes)
    assert len(data) == expected_size
    assert data == b"\0" * expected_size


def test_random_filename():
    """Test random filename generation"""
    size_mb = 5
    filename = random_filename(size_mb)

    assert filename.startswith("5mb-")
    assert filename.endswith(".bytes")
    assert len(filename) == len("5mb-") + 8 + len(".bytes")


def test_successful_file_operations(sync_stats, mock_time, monkeypatch):
    """Test successful file upload, download, and delete operations"""

    def mock_post(*args, **kwargs):
        return MockResponse(200)

    monkeypatch.setattr(requests, "post", mock_post)

    # Test upload
    upload_time = sync_stats.upload_file("test.txt", b"test data")
    assert upload_time == 500.0  # 0.5 seconds * 1000

    # Test download
    download_time = sync_stats.download_file("test.txt")
    assert download_time == 500.0

    # Test delete
    delete_time = sync_stats.delete_file("test.txt")
    assert delete_time == 500.0


def test_file_operation_failures(sync_stats, mock_time, monkeypatch):
    """Test handling of file operation failures"""

    def mock_post(*args, **kwargs):
        return MockResponse(500)

    monkeypatch.setattr(requests, "post", mock_post)

    # Test upload failure
    with pytest.raises(requests.RequestException):
        sync_stats.upload_file("test.txt", b"test data")

    # Test download failure
    with pytest.raises(requests.RequestException):
        sync_stats.download_file("test.txt")

    # Test delete failure (should not raise exception due to ignore_errors=True)
    delete_time = sync_stats.delete_file("test.txt")
    assert delete_time == 500.0


def test_measure_file_transfer_success(sync_stats, mock_time, monkeypatch):
    """Test successful file transfer measurement"""

    def mock_post(*args, **kwargs):
        return MockResponse(200)

    monkeypatch.setattr(requests, "post", mock_post)

    result = sync_stats.measure_file_transfer(1)
    assert isinstance(result, FileTransferDuration)
    assert result.upload == 500.0
    assert result.download == 500.0


def test_measure_file_transfer_failure(sync_stats, mock_time, monkeypatch, capsys):
    """Test failed file transfer measurement"""

    def mock_post(*args, **kwargs):
        raise requests.RequestException("Simulated failure")

    monkeypatch.setattr(requests, "post", mock_post)

    result = sync_stats.measure_file_transfer(1)
    assert result is None

    captured = capsys.readouterr()
    assert "Error during file transfer" in captured.out


def test_get_stats_success(sync_stats, mock_time, monkeypatch):
    """Test successful statistics collection"""

    def mock_post(*args, **kwargs):
        return MockResponse(200)

    monkeypatch.setattr(requests, "post", mock_post)

    stats = sync_stats.get_stats(file_size_mb=1, num_runs=3)
    assert isinstance(stats, DataTransferStats)
    assert stats.file_size_mb == 1
    assert stats.successful_runs == 3
    assert stats.total_runs == 3
    assert isinstance(stats.upload, Stats)
    assert isinstance(stats.download, Stats)


def test_get_stats_all_failures(sync_stats, mock_time, monkeypatch):
    """Test statistics collection with all failures"""

    def mock_post(*args, **kwargs):
        raise requests.RequestException("Simulated failure")

    monkeypatch.setattr(requests, "post", mock_post)
    monkeypatch.setattr(time, "sleep", lambda x: None)  # Skip sleep delay

    with pytest.raises(RuntimeError, match="All .* runs failed"):
        sync_stats.get_stats(file_size_mb=1, num_runs=3)


def test_get_stats_partial_failure(sync_stats, mock_time, monkeypatch):
    """Test statistics collection with some failures"""
    call_count = 0

    def mock_measure_transfer(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # Fail every other attempt
        if call_count % 2 == 0:
            return None
        return FileTransferDuration(upload=500.0, download=500.0)

    monkeypatch.setattr(sync_stats, "measure_file_transfer", mock_measure_transfer)
    monkeypatch.setattr(time, "sleep", lambda x: None)  # Skip sleep delay

    stats = sync_stats.get_stats(file_size_mb=1, num_runs=4)
    assert isinstance(stats, DataTransferStats)
    assert stats.successful_runs == 2  # Half of the runs should succeed
    assert stats.total_runs == 4


def test_get_stats_delay_between_runs(sync_stats, mock_time, monkeypatch):
    """Test delay between runs"""
    calls = []

    def mock_measure_transfer(*args, **kwargs):
        calls.append(1)
        return FileTransferDuration(upload=500.0, download=500.0)

    def mock_sleep(seconds):
        mock_time.sleep_calls.append(seconds)

    monkeypatch.setattr(sync_stats, "measure_file_transfer", mock_measure_transfer)
    monkeypatch.setattr(time, "sleep", mock_sleep)

    sync_stats.get_stats(file_size_mb=1, num_runs=3)

    assert len(mock_time.sleep_calls) == 2
    assert mock_time.sleep_calls == [5, 5]
