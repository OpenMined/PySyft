from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from syft_migration import MigratableObject

from ..config import SyftJobConfig
from ..job_ref import JobRef


class ProtocolCodec(ABC):
    """Interface for reading/writing/listing jobs in one on-disk storage format.

    A codec owns raw disk layout + serialization for the protocol versions it
    lists in ``protocol_versions`` (one codec can serve several — the layout is
    derived from each job's ``protocol_version``). JobStorage owns migration and
    selects a codec by protocol version; a codec never maps back to one. The
    codec carries its own ``version`` independent of the protocols it handles.
    """

    version: str  # the codec's own version ("0", "1", ...)
    protocol_versions: list[str]  # protocol versions this codec reads/writes
    submission_marker = "config.yaml"
    state_marker = "state.yaml"

    def __init__(self, config: SyftJobConfig) -> None:
        self.config = config

    @abstractmethod
    def submission_dir(self, ref: JobRef) -> Path: ...

    @abstractmethod
    def review_dir(self, ref: JobRef) -> Path: ...

    @abstractmethod
    def submission_metadata_path(self, ref: JobRef) -> Path: ...

    @abstractmethod
    def state_path(self, ref: JobRef) -> Path: ...

    @abstractmethod
    def iter_submission_refs(self, datasite_email: str) -> Iterator[JobRef]: ...

    @abstractmethod
    def iter_review_refs(self, datasite_email: str) -> Iterator[JobRef]: ...

    @abstractmethod
    def read(self, path: Path, canonical_name: str) -> dict: ...

    @abstractmethod
    def write(self, path: Path, obj: MigratableObject) -> None: ...
