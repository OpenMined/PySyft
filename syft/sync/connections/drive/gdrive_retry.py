"""Retry logic for Google Drive API calls."""

import http.client
import logging
import random
import socket
import ssl
import time
from typing import Any, TypeVar

from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 8.0  # seconds
DEFAULT_BACKOFF_MULTIPLIER = 2.0

RETRYABLE_STATUS_CODES = {500, 502, 503, 429}
RETRYABLE_REASONS = {
    "internalError",
    "backendError",
    "rateLimitExceeded",
    "userRateLimitExceeded",
    "badRequest",  # GDrive resumable uploads can return transient 400 "badRequest"
}

# Socket/TLS transients raised by httplib2 when Google closes an idle keepalive
# connection. googleapiclient's own retry loop only catches ssl.SSLError and
# socket.timeout, so these escape and surface to callers as unhandled errors.
RETRYABLE_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
    TimeoutError,
    socket.timeout,
    ssl.SSLError,
    http.client.RemoteDisconnected,
    http.client.BadStatusLine,
)

T = TypeVar("T")


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is transient and should be retried."""
    if isinstance(error, RETRYABLE_TRANSPORT_ERRORS):
        return True
    if isinstance(error, HttpError):
        if error.resp.status in RETRYABLE_STATUS_CODES:
            return True
        # Check error reason in response body
        if hasattr(error, "error_details") and error.error_details:
            for detail in error.error_details:
                if detail.get("reason") in RETRYABLE_REASONS:
                    return True
    return False


def _describe(error: Exception) -> str:
    if isinstance(error, HttpError):
        return str(error.resp.status)
    return type(error).__name__


def execute_with_retries(
    request: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
) -> T:
    """Execute a Google Drive API request with retry on transient errors.

    Uses exponential backoff with jitter.

    Args:
        request: A Google API request object (has .execute() method)
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_multiplier: Multiplier for exponential backoff

    Returns:
        The result of the API call

    Raises:
        HttpError: If the request fails after all retries or with non-retryable error
    """
    last_error = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return request.execute()
        except (HttpError, *RETRYABLE_TRANSPORT_ERRORS) as e:
            last_error = e
            if not is_retryable_error(e) or attempt == max_retries:
                raise

            # Add jitter (+-20%)
            jitter = delay * 0.2 * (random.random() * 2 - 1)
            sleep_time = min(delay + jitter, max_delay)

            logger.warning(
                f"Google Drive API error (attempt {attempt + 1}/{max_retries + 1}): "
                f"{_describe(e)} - retrying in {sleep_time:.2f}s"
            )
            time.sleep(sleep_time)
            delay = min(delay * backoff_multiplier, max_delay)

    raise last_error


def next_chunk_with_retries(
    downloader: MediaIoBaseDownload,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
) -> tuple[Any, bool]:
    """Execute next_chunk() on a downloader with retry on transient errors.

    Args:
        downloader: A MediaIoBaseDownload instance
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_multiplier: Multiplier for exponential backoff

    Returns:
        Tuple of (status, done) from next_chunk()

    Raises:
        HttpError: If the download fails after all retries or with non-retryable error
    """
    last_error = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return downloader.next_chunk()
        except (HttpError, *RETRYABLE_TRANSPORT_ERRORS) as e:
            last_error = e
            if not is_retryable_error(e) or attempt == max_retries:
                raise

            # Add jitter (+-20%)
            jitter = delay * 0.2 * (random.random() * 2 - 1)
            sleep_time = min(delay + jitter, max_delay)

            logger.warning(
                f"Google Drive download error (attempt {attempt + 1}/{max_retries + 1}): "
                f"{_describe(e)} - retrying in {sleep_time:.2f}s"
            )
            time.sleep(sleep_time)
            delay = min(delay * backoff_multiplier, max_delay)

    raise last_error


def batch_execute_with_retries(
    batch: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
) -> None:
    """Execute a batch request with retry on transient errors.

    Args:
        batch: A BatchHttpRequest instance
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_multiplier: Multiplier for exponential backoff

    Raises:
        HttpError: If the batch fails after all retries or with non-retryable error
    """
    last_error = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            batch.execute()
            return
        except (HttpError, *RETRYABLE_TRANSPORT_ERRORS) as e:
            last_error = e
            if not is_retryable_error(e) or attempt == max_retries:
                raise

            # Add jitter (+-20%)
            jitter = delay * 0.2 * (random.random() * 2 - 1)
            sleep_time = min(delay + jitter, max_delay)

            logger.warning(
                f"Google Drive batch error (attempt {attempt + 1}/{max_retries + 1}): "
                f"{_describe(e)} - retrying in {sleep_time:.2f}s"
            )
            time.sleep(sleep_time)
            delay = min(delay * backoff_multiplier, max_delay)

    raise last_error
