from pathlib import Path
from typing import Iterator

import yaml
from syft_migration import MigratableObject

from ..config import (
    METADATA_FILENAME,
    PRIVATE_METADATA_FILENAME,
    SYFT_DATASETS_FOLDER_NAME,
    SyftBoxConfig,
    protocol_dir_name,
)
from ..dataset_ref import DatasetRef
from .base import ProtocolCodec


class DatasetConfigV1:
    """Versioned layout (>= 0.1.21): datasets under syft_datasets/v<n>/, identity fields on disk.

    Serves every protocol version in ``protocol_versions``. The v<n> segment is
    derived from the protocol version (of the ref, or of each version when
    scanning), never hardcoded, so a protocol bump that doesn't change the layout
    is just appending the new version to ``protocol_versions``.
    """

    version = "1"
    protocol_versions = ["1"]
    datasets_folder_name = SYFT_DATASETS_FOLDER_NAME
    metadata_filename = METADATA_FILENAME
    private_metadata_filename = PRIVATE_METADATA_FILENAME

    def __init__(self, syftbox_config: SyftBoxConfig) -> None:
        self.syftbox_config = syftbox_config

    def public_all_dataset_folder(self, datasite: str) -> Path:
        return (
            self.syftbox_config.datasite_public_root(datasite)
            / self.datasets_folder_name
        )

    def private_all_datatset_folder(self, owner: str) -> Path:
        return (
            self.syftbox_config.datasite_private_root(owner) / self.datasets_folder_name
        )

    def _segment(self, ref: DatasetRef) -> str:
        # Derived from the ref, never hardcoded. This codec only handles
        # versioned (segmented) protocols, so a ref that resolves to no segment
        # (e.g. protocol 0) must never reach here.
        segment = protocol_dir_name(ref.protocol_version)
        if segment is None:
            raise ValueError(
                "DatasetConfigV1 cannot resolve a path for protocol version "
                f"{ref.protocol_version!r}: it has no versioned segment"
            )
        return segment

    def public_dataset_dir(self, ref: DatasetRef) -> Path:
        return self.public_all_dataset_folder(ref.owner) / self._segment(ref) / ref.name

    def private_dataset_dir(self, ref: DatasetRef) -> Path:
        return (
            self.private_all_datatset_folder(ref.owner) / self._segment(ref) / ref.name
        )

    def metadata_path(self, ref: DatasetRef) -> Path:
        return self.public_dataset_dir(ref) / self.metadata_filename

    def private_metadata_path(self, ref: DatasetRef) -> Path:
        return self.private_dataset_dir(ref) / self.private_metadata_filename

    def iter_dataset_refs_all_supported_protocols(
        self, datasite_email: str
    ) -> Iterator[DatasetRef]:
        """Yield refs for datasets stored under any supported protocol's segment."""
        root = self.public_all_dataset_folder(datasite_email)
        for protocol_version in self.protocol_versions:
            version_dir = root / protocol_dir_name(protocol_version)
            if not version_dir.is_dir():
                continue
            for entry in sorted(p for p in version_dir.iterdir() if p.is_dir()):
                if (entry / self.metadata_filename).exists():
                    yield DatasetRef(datasite_email, entry.name, protocol_version)


class ProtocolCodecV1(ProtocolCodec):
    """Reads/writes the versioned layout with identity fields intact."""

    dataset_config_cls = DatasetConfigV1

    def iter_dataset_refs(self, datasite_email: str) -> Iterator[DatasetRef]:
        return self.dataset_config.iter_dataset_refs_all_supported_protocols(
            datasite_email
        )

    def read(self, path: Path, canonical_name: str) -> dict:
        # Files already carry canonical_name/version on disk.
        return yaml.safe_load(path.read_text()) or {}

    def write(self, path: Path, obj: MigratableObject) -> None:
        data = obj.disk_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, indent=2, sort_keys=False))
