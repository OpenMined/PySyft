from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from enum import Enum, IntEnum
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator
from pydantic import ValidationError as PydanticValidationError
from syft_core.types import PathLike, to_path
from syft_core.url import SyftBoxURL
from typing_extensions import (
    ClassVar,
    Dict,
    List,
    Optional,
    Self,
    Type,
    TypeAlias,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

# Type aliases for better readability
JSONPrimitive: TypeAlias = Union[str, int, float, bool, None]
JSONValue: TypeAlias = Union[Dict[str, "JSONValue"], List["JSONValue"], JSONPrimitive]
JSON: TypeAlias = Union[str, bytes, bytearray]
Headers: TypeAlias = dict[str, str]
PYDANTIC = TypeVar("PYDANTIC", bound=BaseModel)


# Constants
DEFAULT_MESSAGE_EXPIRY: int = 60 * 60 * 24  # 1 days in seconds
DEFAULT_POLL_INTERVAL: float = 0.1
DEFAULT_TIMEOUT: float = 300  # 5 minutes in seconds


def validate_syftbox_url(url: Union[SyftBoxURL, str]) -> SyftBoxURL:
    if isinstance(url, str):
        return SyftBoxURL(url)
    if isinstance(url, SyftBoxURL):
        return url
    raise ValueError(f"Invalid type for url: {type(url)}. Expected str or SyftBoxURL.")


class SyftMethod(str, Enum):
    """HTTP methods supported by the Syft protocol."""

    GET = "GET"
    HEAD = "HEAD"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class SyftStatus(IntEnum):
    """Standard HTTP-like status codes for Syft responses."""

    SYFT_200_OK = 200
    SYFT_400_BAD_REQUEST = 400
    SYFT_403_FORBIDDEN = 403
    SYFT_404_NOT_FOUND = 404
    SYFT_419_EXPIRED = 419
    SYFT_500_SERVER_ERROR = 500

    @property
    def is_success(self) -> bool:
        """Check if the status code indicates success."""
        return 200 <= self.value < 300

    @property
    def is_error(self) -> bool:
        """Check if the status code indicates an error."""
        return self.value >= 400


class Base(BaseModel):
    """Base model with enhanced serialization capabilities."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat(),
        },
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    def dumps(self) -> str:
        """Serialize the model instance to JSON formatted str.

        Returns:
            JSON string representation of the model instance.

        Raises:
            pydantic.ValidationError: If the model contains invalid data.
            TypeError: If the model contains types that cannot be JSON serialized.
        """
        return self.model_dump_json()

    def dump(self, path: PathLike) -> None:
        """Serialize the model instance as JSON to a file.

        Args:
            path: The file path where the JSON data will be written.

        Raises:
            pydantic.ValidationError: If the model contains invalid data.
            TypeError: If the model contains types that cannot be JSON serialized.
            PermissionError: If lacking permission to write to the path.
            OSError: If there are I/O related errors.
            FileNotFoundError: If the parent directory doesn't exist.
        """
        to_path(path).write_text(self.dumps())

    @classmethod
    def loads(cls, data: JSON) -> Self:
        """Load a model instance from a JSON string or bytes.

        Args:
            data: JSON data to parse. Can be string or binary data.

        Returns:
            A new instance of the model class.

        Raises:
            pydantic.ValidationError: If JSON doesn't match the model's schema.
            ValueError: If the input is not valid JSON.
            TypeError: If input type is not str, bytes, or bytearray.
            UnicodeDecodeError: If binary input cannot be decoded as UTF-8.
        """
        return cls.model_validate_json(data)

    @classmethod
    def load(cls, path: PathLike) -> Self:
        """Load a model instance from a JSON file.

        Args:
            path: Path to the JSON file to read.

        Returns:
            A new instance of the model class.

        Raises:
            pydantic.ValidationError: If JSON doesn't match the model's schema.
            ValueError: If file content is not valid JSON.
            FileNotFoundError: If the file doesn't exist.
            PermissionError: If lacking permission to read the file.
            OSError: If there are I/O related errors.
            UnicodeDecodeError: If content cannot be decoded as UTF-8.
        """
        return cls.loads(to_path(path).read_text())


class SyftMessage(Base):
    """Base message class for Syft protocol communication."""

    VERSION: ClassVar[int] = 1

    id: UUID = Field(default_factory=uuid4)
    """Unique identifier of the message."""

    sender: str
    """The sender of the message."""

    url: SyftBoxURL
    """The URL of the message."""

    body: Optional[bytes] = None
    """The body of the message in bytes."""

    headers: Headers = Field(default_factory=dict)
    """Additional headers for the message."""

    created: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    """Timestamp when the message was created."""

    expires: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
        + timedelta(seconds=DEFAULT_MESSAGE_EXPIRY)
    )
    """Timestamp when the message expires."""

    @property
    def age(self) -> float:
        """Return the age of the message in seconds."""
        return (datetime.now(timezone.utc) - self.created).total_seconds()

    @property
    def is_expired(self) -> bool:
        """Check if the message has expired."""
        return datetime.now(timezone.utc) > self.expires

    @field_validator("url", mode="before")
    @classmethod
    def validate_url(cls, value) -> SyftBoxURL:
        return validate_syftbox_url(value)

    def get_message_id(self) -> UUID:
        """Generate a deterministic UUID from the message contents."""
        return UUID(bytes=self.__msg_hash().digest()[:16], version=4)

    def get_message_hash(self) -> str:
        """Generate a hash of the message contents."""
        return self.__msg_hash().hexdigest()

    def __msg_hash(self):
        """Generate a hash of the message contents."""
        m = self.model_dump_json(include={"url", "method", "sender", "headers", "body"})
        return hashlib.sha256(m.encode())

    def text(self) -> str:
        """Decode the body as a string.

        Args:
            encoding: Character encoding to use for decoding bytes. Defaults to "utf-8".

        Returns:
            Decoded string representation of the body.

        Raises:
            UnicodeDecodeError: If bytes cannot be decoded with specified encoding
        """
        if not self.body:
            return ""
        return self.body.decode()

    def json(self, **kwargs) -> JSONValue:
        """Parse bytes body into JSON data.

        Args:
            encoding: Character encoding to use for decoding bytes. Defaults to "utf-8".

        Returns:
            Parsed JSON data as dict, list, or primitive value.

        Raises:
            json.JSONDecodeError: If body contains invalid JSON
            UnicodeDecodeError: If bytes cannot be decoded with specified encoding
        """
        return json.loads(self.text())

    def model(self, model_cls: Type[PYDANTIC]) -> PYDANTIC:
        """Parse JSON body into a Pydantic model instance.

        Args:
            model_cls: A Pydantic model class to parse the JSON into

        Returns:
            An instance of the provided model class

        Raises:
            ValidationError: If JSON data doesn't match model schema
        """

        return model_cls.model_validate_json(self.body)


class SyftError(Exception):
    """Base exception for Syft-related errors."""

    pass


class SyftTimeoutError(SyftError):
    """Raised when a request times out."""

    pass


class SyftRequest(SyftMessage):
    """Request message in the Syft protocol."""

    method: SyftMethod = SyftMethod.GET


class SyftResponse(SyftMessage):
    """Response message in the Syft protocol."""

    status_code: SyftStatus = SyftStatus.SYFT_200_OK

    @property
    def is_success(self) -> bool:
        """Check if the response indicates success."""
        return self.status_code.is_success

    def raise_for_status(self):
        if self.status_code.is_error:
            raise SyftError(
                f"Request failed with status code {self.status_code}. Reason: {self.body}"
            )

    @classmethod
    def system_response(cls, status_code: SyftStatus, message: str) -> Self:
        return cls(
            status_code=status_code,
            body=message.encode(),
            url=SyftBoxURL("syft://system@syftbox.localhost"),
            sender="system@syftbox.localhost",
        )


class SyftFuture(Base):
    """Represents an asynchronous Syft RPC operation on a file system transport.

    Attributes:
        id: Identifier of the corresponding request and response.
        path: Path where request and response files are stored.
        expires: Timestamp when the request expires.
    """

    id: UUID
    """Identifier of the corresponding request and response."""

    path: Path
    """Path where request and response files are stored"""

    expires: datetime
    """Timestamp when the request expires"""

    _request: Optional[SyftRequest] = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._request = data.get("request")
        if not self._request:
            self._request = SyftRequest.load(self.request_path)

    @property
    def request_path(self) -> Path:
        """Path to the request file."""
        return to_path(self.path) / f"{self.id}.request"

    @property
    def response_path(self) -> Path:
        """Path to the response file."""
        return to_path(self.path) / f"{self.id}.response"

    @property
    def rejected_path(self) -> Path:
        """Path to the rejected request marker file."""
        return self.request_path.with_suffix(f".syftrejected{self.request_path.suffix}")

    @property
    def is_rejected(self) -> bool:
        """Check if the request has been rejected."""
        return self.rejected_path.exists()

    @property
    def is_expired(self) -> bool:
        """Check if the future has expired."""
        return datetime.now(timezone.utc) > self.expires

    @property
    def request(self) -> SyftRequest:
        """Get the underlying request object."""

        if not self._request:
            self._request = SyftRequest.load(self.request_path)
        return self._request

    def wait(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> SyftResponse:
        """Wait for the future to complete and return the Response.

        Args:
            timeout: Maximum time to wait in seconds. None means wait until the request expires.
            poll_interval: Time in seconds between polling attempts.

        Returns:
            The response object.

        Raises:
            SyftTimeoutError: If timeout is reached before receiving a response.
            ValueError: If timeout or poll_interval is negative.
        """
        if timeout is not None and timeout <= 0:
            raise ValueError("Timeout must be greater than 0")
        if poll_interval <= 0:
            raise ValueError("Poll interval must be greater than 0")

        deadline = time.monotonic() + (timeout or float("inf"))

        while time.monotonic() < deadline:
            try:
                response = self.resolve()
                if response is not None:
                    return response
                time.sleep(poll_interval)
            except Exception as e:
                logger.error(f"Error while resolving future: {str(e)}")
                raise

        raise SyftTimeoutError(
            f"Timeout reached after waiting {timeout} seconds for response"
        )

    def resolve(self) -> Optional[SyftResponse]:
        """Attempt to resolve the future to a response.

        Returns:
            The response if available, None if still pending.
        """

        # Check for rejection first
        if self.is_rejected:
            self.request_path.unlink(missing_ok=True)
            self.rejected_path.unlink(missing_ok=True)
            return SyftResponse.system_response(
                status_code=SyftStatus.SYFT_403_FORBIDDEN,
                message="Request was rejected by the SyftBox cache server due to permissions issue",
            )

        # Check for existing response
        if self.response_path.exists():
            return self._handle_existing_response()

        # If both request and response are missing, the request has expired
        # and they got cleaned up by the server.
        if not self.request_path.exists():
            return SyftResponse.system_response(
                status_code=SyftStatus.SYFT_404_NOT_FOUND,
                message=f"Request with {self.id} not found",
            )

        # Check for expired request
        request = SyftRequest.load(self.request_path)
        if request.is_expired:
            self.request_path.unlink(missing_ok=True)
            self.response_path.unlink(missing_ok=True)
            return SyftResponse.system_response(
                status_code=SyftStatus.SYFT_419_EXPIRED,
                message=f"Request with {self.id} expired on {request.expires}",
            )

        # No response yet
        return None

    def _handle_existing_response(self) -> SyftResponse:
        """Process an existing response file.

        Returns:
            The loaded response object.

        Note:
            If the response file exists but is invalid or expired,
            returns an appropriate error response instead of raising an exception.
        """
        try:
            response = SyftResponse.load(self.response_path)
            # preserve results, but change status code to 419
            if response.is_expired:
                response.status_code = SyftStatus.SYFT_419_EXPIRED
            return response
        except (PydanticValidationError, ValueError, UnicodeDecodeError) as e:
            logger.error(f"Error loading response: {str(e)}")
            return SyftResponse.system_response(
                status_code=SyftStatus.SYFT_500_SERVER_ERROR,
                message=f"Error loading response: {str(e)}",
            )
        finally:
            self.request_path.unlink(missing_ok=True)
            self.response_path.unlink(missing_ok=True)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, SyftFuture):
            return False
        return self.id == other.id


class SyftBulkFuture(Base):
    futures: List[SyftFuture]
    responses: List[SyftResponse] = []

    def resolve(self) -> None:
        """Resolve all futures and store the responses."""
        for future in self.pending:
            if response := future.resolve():
                self.responses.append(response)

    def gather_completed(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> List[SyftResponse]:
        """Wait for all futures to complete and return a list of responses.

        Returns a list of responses in the order of the futures list. If a future
        times out, it will be omitted from the list. If the timeout is reached before
        all futures complete, the function will return the responses received so far.

        Args:
            timeout: Maximum time to wait in seconds.
            poll_interval: Time in seconds between polling attempts.
        Returns:
            A list of response objects.
        Raises:
            ValueError: If timeout or poll_interval is negative.
        """
        if timeout is not None and timeout <= 0:
            raise ValueError("Timeout must be greater than 0")
        if poll_interval <= 0:
            raise ValueError("Poll interval must be greater than 0")

        deadline = time.monotonic() + (timeout or float("inf"))

        while time.monotonic() < deadline:
            self.resolve()
            if not self.pending:
                logger.debug("All futures have resolved")
                break
            time.sleep(poll_interval)

        return self.responses

    @property
    def id(self) -> UUID:
        """Generate a deterministic UUID from all future IDs.

        Returns:
            A single UUID derived from hashing all future IDs.
        """
        # Combine all UUIDs and hash them
        combined = ",".join(str(f.id) for f in self.futures)
        hash_bytes = hashlib.sha256(combined.encode()).digest()[:16]
        # Use first 16 bytes of hash to create a new UUID
        return UUID(bytes=hash_bytes, version=4)

    @property
    def pending(self) -> List[SyftFuture]:
        """Return a list of futures that have not yet resolved."""
        completed = {r.id for r in self.responses}
        return [f for f in self.futures if f.id not in completed]

    @property
    def failures(self) -> List[SyftResponse]:
        """Return a list of failed responses."""
        return [r for r in self.responses if not r.is_success]

    @property
    def successes(self) -> List[SyftResponse]:
        """Return a list of successful responses."""
        return [r for r in self.responses if r.is_success]

    @property
    def all_failed(self) -> bool:
        """Check if all futures have failed."""
        return len(self.failures) == len(self.futures)
