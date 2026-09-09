from pathlib import Path
from typing import Iterator

import yaml
from syft_migration import MigratableObject

from ..config import (
    METADATA_FILENAME,
    PRIVATE_METADATA_FILENAME,
    SYFT_DATASETS_FOLDER_NAME,
    SyftBoxConfig,
    is_protocol_dir_name,
)
from ..dataset_ref import DatasetRef
from .base import ProtocolCodec


class DatasetConfigV0:
    """Pre-versioning layout (<= 0.1.20): datasets flat under syft_datasets/, no segment."""

    version = "0"
    protocol_versions = ["0"]
    datasets_folder_name = SYFT_DATASETS_FOLDER_NAME
    metadata_filename = METADATA_FILENAME
    private_metadata_filename = PRIVATE_METADATA_FILENAME

    def __init__(self, syftbox_config: SyftBoxConfig) -> None:
        self.syftbox_config = syftbox_config

    def public_all_datasets_folder(self, datasite: str) -> Path:
        return (
            self.syftbox_config.datasite_public_root(datasite)
            / self.datasets_folder_name
        )

    def private_all_datasets_folder(self, owner: str) -> Path:
        return (
            self.syftbox_config.datasite_private_root(owner) / self.datasets_folder_name
        )

    def public_dataset_dir(self, ref: DatasetRef) -> Path:
        return self.public_all_datasets_folder(ref.owner) / ref.name

    def private_dataset_dir(self, ref: DatasetRef) -> Path:
        return self.private_all_datasets_folder(ref.owner) / ref.name

    def metadata_path(self, ref: DatasetRef) -> Path:
        return self.public_dataset_dir(ref) / self.metadata_filename

    def private_metadata_path(self, ref: DatasetRef) -> Path:
        return self.private_dataset_dir(ref) / self.private_metadata_filename

    def iter_dataset_refs(self, datasite_email: str) -> Iterator[DatasetRef]:
        root = self.public_all_datasets_folder(datasite_email)
        if not root.is_dir():
            return
        for entry in sorted(p for p in root.iterdir() if p.is_dir()):
            # v<n>/ segments belong to a versioned layout, not the flat one.
            if is_protocol_dir_name(entry.name):
                continue
            if (entry / self.metadata_filename).exists():
                yield DatasetRef(datasite_email, entry.name, "0")


class ProtocolCodecV0(ProtocolCodec):
    """Reads/writes the flat layout, stripping the identity fields off disk."""

    dataset_config_cls = DatasetConfigV0

    def read(self, path: Path, canonical_name: str) -> dict:
        data = yaml.safe_load(path.read_text()) or {}
        # Protocol-0 files predate the identity fields; they are all version 1.
        data.setdefault("canonical_name", canonical_name)
        data.setdefault("version", "1")
        return data

    def write(self, path: Path, obj: MigratableObject) -> None:
        data = obj.disk_dict()
        # Byte-match the pre-versioning (<= 0.1.20) on-disk format.
        data.pop("canonical_name", None)
        data.pop("version", None)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, indent=2, sort_keys=False))
