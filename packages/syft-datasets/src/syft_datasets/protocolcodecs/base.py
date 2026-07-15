from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from syft_migration import MigratableObject

from ..config import METADATA_FILENAME, PRIVATE_METADATA_FILENAME, SyftBoxConfig
from ..dataset_ref import DatasetRef


class ProtocolCodec(ABC):
    """Interface for reading/writing/listing datasets in one on-disk storage format.

    A codec owns raw disk layout + serialization for the protocol versions it
    lists in ``protocol_versions`` (one codec can serve several — the layout is
    derived from each dataset's ``protocol_version``). DatasetStorage owns
    migration and selects a codec by protocol version; a codec never maps back
    to one. The codec carries its own ``version`` independent of the protocols
    it handles.

    A dataset is a directory with a public ``dataset.yaml`` (the discovery
    marker, synced to peers) plus mock files, and a separate
    ``private_metadata.yaml`` under the owner's private/ tree (never synced).
    """

    version: str  # the codec's own version ("0", "1", ...)
    protocol_versions: list[str]  # protocol versions this codec reads/writes
    metadata_marker = METADATA_FILENAME
    private_metadata_marker = PRIVATE_METADATA_FILENAME

    def __init__(self, config: SyftBoxConfig) -> None:
        self.config = config

    @abstractmethod
    def public_dataset_dir(self, ref: DatasetRef) -> Path: ...

    @abstractmethod
    def private_dataset_dir(self, ref: DatasetRef) -> Path: ...

    def metadata_path(self, ref: DatasetRef) -> Path:
        return self.public_dataset_dir(ref) / self.metadata_marker

    def private_metadata_path(self, ref: DatasetRef) -> Path:
        return self.private_dataset_dir(ref) / self.private_metadata_marker

    @abstractmethod
    def iter_dataset_refs(self, datasite_email: str) -> Iterator[DatasetRef]: ...

    @abstractmethod
    def read(self, path: Path, canonical_name: str) -> dict: ...

    @abstractmethod
    def write(self, path: Path, obj: MigratableObject) -> None: ...
