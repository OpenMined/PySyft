"""Unit tests for syft_client.sync.connections.drive.gdrive_retry.

Focus: the retry helpers must recover from socket-level transients
(ConnectionResetError, BrokenPipeError, etc.) that surface when httplib2's
cached keepalive connection is closed server-side. Pre-fix these escaped the
retry loop entirely; this test pins the behavior so it doesn't regress.

The three retry helpers (execute / next_chunk / batch) share the same
`except (HttpError, *RETRYABLE_TRANSPORT_ERRORS)` clause, so the
parametrized registry test in `is_retryable_error` is what actually proves
each error type is recognized. The per-helper tests are smoke tests to
confirm the wiring, not exhaustive cross-products.
"""

import http.client
import socket
import ssl
from unittest.mock import Mock, patch

import pytest
from googleapiclient.errors import HttpError

from syft_client.sync.connections.drive.gdrive_retry import (
    batch_execute_with_retries,
    execute_with_retries,
    is_retryable_error,
    next_chunk_with_retries,
)


def _http_error(status: int) -> HttpError:
    resp = Mock()
    resp.status = status
    resp.reason = "test"
    return HttpError(resp, b"{}")


TRANSPORT_INSTANCES = [
    ConnectionResetError("peer reset"),
    ConnectionAbortedError("aborted"),
    BrokenPipeError("epipe"),
    TimeoutError("timed out"),
    socket.timeout("sock timeout"),
    ssl.SSLError("ssl"),
    http.client.RemoteDisconnected("remote disconnected"),
    http.client.BadStatusLine("bad status"),
]


@pytest.fixture(autouse=True)
def _no_sleep():
    """Patch time.sleep so tests run instantly."""
    with patch("syft_client.sync.connections.drive.gdrive_retry.time.sleep"):
        yield


# ---------- is_retryable_error registry -------------------------------------
# These are the safety net: they prove every error class we claim to handle
# is actually in the retryable set. If someone removes a class from
# RETRYABLE_TRANSPORT_ERRORS, the matching case here fails by name.


@pytest.mark.parametrize("err", TRANSPORT_INSTANCES)
def test_is_retryable_error_transport(err):
    assert is_retryable_error(err) is True


def test_is_retryable_error_retryable_http_statuses():
    for status in (500, 502, 503, 429):
        assert is_retryable_error(_http_error(status)) is True, status


def test_is_retryable_error_non_retryable_http_statuses():
    for status in (400, 403, 404):
        assert is_retryable_error(_http_error(status)) is False, status


def test_is_retryable_error_unrelated_exception():
    assert is_retryable_error(ValueError("nope")) is False


# ---------- execute_with_retries --------------------------------------------
# This is the hot path -- most Drive calls flow through here. The exhaust /
# immediate-raise / pass-through behaviors are covered here once; the other
# two helpers use the same loop structure and only get smoke tests.


def test_execute_with_retries_recovers_from_transport_error():
    request = Mock()
    request.execute.side_effect = [ConnectionResetError("reset"), {"ok": True}]

    result = execute_with_retries(request, initial_delay=0, max_delay=0)

    assert result == {"ok": True}
    assert request.execute.call_count == 2


def test_execute_with_retries_exhausts_then_raises():
    request = Mock()
    request.execute.side_effect = ConnectionResetError("peer reset")

    with pytest.raises(ConnectionResetError):
        execute_with_retries(request, max_retries=2, initial_delay=0, max_delay=0)

    # max_retries=2 means 1 initial attempt + 2 retries = 3 total.
    assert request.execute.call_count == 3


def test_execute_with_retries_non_retryable_http_raises_immediately():
    request = Mock()
    request.execute.side_effect = _http_error(404)

    with pytest.raises(HttpError):
        execute_with_retries(request, initial_delay=0, max_delay=0)

    assert request.execute.call_count == 1


def test_execute_with_retries_retries_retryable_http():
    request = Mock()
    request.execute.side_effect = [_http_error(503), {"ok": True}]

    result = execute_with_retries(request, initial_delay=0, max_delay=0)

    assert result == {"ok": True}
    assert request.execute.call_count == 2


def test_execute_with_retries_passes_through_unrelated_exception():
    request = Mock()
    request.execute.side_effect = ValueError("nope")

    with pytest.raises(ValueError):
        execute_with_retries(request, initial_delay=0, max_delay=0)

    assert request.execute.call_count == 1


# ---------- next_chunk_with_retries / batch_execute_with_retries ------------
# Smoke tests. Same retry loop as execute_with_retries; we only need to
# confirm the helper actually wires the right method (next_chunk / execute)
# to the loop.


def test_next_chunk_with_retries_recovers():
    downloader = Mock()
    downloader.next_chunk.side_effect = [BrokenPipeError("epipe"), (Mock(), True)]

    _, done = next_chunk_with_retries(downloader, initial_delay=0, max_delay=0)

    assert done is True
    assert downloader.next_chunk.call_count == 2


def test_batch_execute_with_retries_recovers():
    batch = Mock()
    batch.execute.side_effect = [ConnectionResetError("reset"), None]

    batch_execute_with_retries(batch, initial_delay=0, max_delay=0)

    assert batch.execute.call_count == 2
