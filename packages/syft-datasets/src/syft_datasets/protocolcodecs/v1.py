from pathlib import Path
from typing import Iterator

import yaml
from syft_migration import MigratableObject

from ..config import (
    METADATA_FILENAME,
    PRIVATE_METADATA_FILENAME,
    SYFT_DATASETS_FOLDER_NAME,
    SyftBoxConfig,
)
from ..dataset_ref import DatasetRef
from .base import ProtocolCodec


class DatasetConfigV1:
    """Versioned layout (>= 0.1.21): datasets under syft_datasets/v1/, identity fields on disk."""

    version = "1"
    protocol_versions = ["1"]
    protocol_segment = "v1"
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

    def public_dataset_dir(self, ref: DatasetRef) -> Path:
        return (
            self.public_all_dataset_folder(ref.owner) / self.protocol_segment / ref.name
        )

    def private_dataset_dir(self, ref: DatasetRef) -> Path:
        return (
            self.private_all_datatset_folder(ref.owner)
            / self.protocol_segment
            / ref.name
        )

    def metadata_path(self, ref: DatasetRef) -> Path:
        return self.public_dataset_dir(ref) / self.metadata_filename

    def private_metadata_path(self, ref: DatasetRef) -> Path:
        return self.private_dataset_dir(ref) / self.private_metadata_filename

    def iter_dataset_refs(self, datasite_email: str) -> Iterator[DatasetRef]:
        version_dir = (
            self.public_all_dataset_folder(datasite_email) / self.protocol_segment
        )
        if not version_dir.is_dir():
            return
        for entry in sorted(p for p in version_dir.iterdir() if p.is_dir()):
            if (entry / self.metadata_filename).exists():
                yield DatasetRef(datasite_email, entry.name, "1")


class ProtocolCodecV1(ProtocolCodec):
    """Reads/writes the versioned layout with identity fields intact."""

    dataset_config_cls = DatasetConfigV1

    def read(self, path: Path, canonical_name: str) -> dict:
        # Files already carry canonical_name/version on disk.
        return yaml.safe_load(path.read_text()) or {}

    def write(self, path: Path, obj: MigratableObject) -> None:
        data = obj.disk_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, indent=2, sort_keys=False))
