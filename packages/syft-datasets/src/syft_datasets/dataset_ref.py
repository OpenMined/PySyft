from dataclasses import dataclass


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a dataset's metadata (dataset.yaml) does not exist."""


class PrivateConfigNotFoundError(FileNotFoundError):
    """Raised when a dataset's private_metadata.yaml does not exist."""


@dataclass(frozen=True)
class DatasetRef:
    """One dataset on disk: who owns it and its protocol layout.

    A dataset is single-owner and broadcast-read (one public copy read by the
    whole audience), so there is no per-reader field: identity is
    (owner, name, protocol_version).
    """

    owner: str  # datasite email of the DO who owns the dataset
    name: str
    protocol_version: str  # "0" (no path segment) or "1"+ (v<n> segment)
