class BlobStorageException(Exception):
    """Base exception class for blob storage errors."""


class BlobStorageInvalidCredentialsError(BlobStorageException):
    """Invalid credentials provided."""


class BlobStorageAuthenticationError(BlobStorageException):
    """Authentication failed."""


class BlobStoragePermissionDeniedError(BlobStorageException):
    """Insufficient permissions."""


class BlobStorageClientError(BlobStorageException):
    """Client encountered an error."""


class BlobStorageNotFoundError(BlobStorageException):
    """Blob storage bucket not found."""


class BlobStorageAllocationError(BlobStorageException):
    """Failed to allocate storage."""


class BlobStorageReadError(BlobStorageException):
    """Failed to read from blob storage."""


class BlobStorageWriteError(BlobStorageException):
    """Failed to write to blob storage."""


class BlobStorageDeleteError(BlobStorageException):
    """Failed to delete blob."""
