from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from syft_migration import MigratableObject

from ..config import SyftBoxConfig
from ..dataset_ref import DatasetRef


class ProtocolCodec(ABC):
    """Reads/writes dataset metadata for one on-disk layout.

    The codec owns serialization (``read``/``write``); its ``dataset_config`` owns
    disk layout (paths, filenames, iteration) and nests the SyftBoxConfig it
    resolves against. DatasetStorage owns migration and selects a codec by
    protocol version. Path/iteration methods here delegate to the config, so
    DatasetStorage keeps a single object to talk to.
    """

    # Each concrete codec sets its DatasetConfigV<n> class (see v0.py / v1.py).
    dataset_config_cls: type

    def __init__(self, config: SyftBoxConfig) -> None:
        self.dataset_config = self.dataset_config_cls(config)

    @property
    def version(self) -> str:
        return self.dataset_config.version

    @property
    def protocol_versions(self) -> list[str]:
        return self.dataset_config.protocol_versions

    def public_dataset_dir(self, ref: DatasetRef) -> Path:
        return self.dataset_config.public_dataset_dir(ref)

    def private_dataset_dir(self, ref: DatasetRef) -> Path:
        return self.dataset_config.private_dataset_dir(ref)

    def metadata_path(self, ref: DatasetRef) -> Path:
        return self.dataset_config.metadata_path(ref)

    def private_metadata_path(self, ref: DatasetRef) -> Path:
        return self.dataset_config.private_metadata_path(ref)

    def iter_dataset_refs(self, datasite_email: str) -> Iterator[DatasetRef]:
        return self.dataset_config.iter_dataset_refs(datasite_email)

    @abstractmethod
    def read(self, path: Path, canonical_name: str) -> dict: ...

    @abstractmethod
    def write(self, path: Path, obj: MigratableObject) -> None: ...
