from pathlib import Path
from typing import Iterator

import yaml
from syft_migration import MigratableObject

from ..config import is_protocol_dir_name
from ..dataset_ref import DatasetRef
from .base import ProtocolCodec


class ProtocolCodecV0(ProtocolCodec):
    """Pre-versioning layout (<= 0.1.20): datasets flat under syft_datasets/, no identity fields."""

    version = "0"
    protocol_versions = ["0"]

    def public_dataset_dir(self, ref: DatasetRef) -> Path:
        return self.config.get_mock_dataset_dir(
            ref.name, ref.owner, ref.protocol_version
        )

    def private_dataset_dir(self, ref: DatasetRef) -> Path:
        return self.config.get_private_dataset_dir(
            ref.owner, ref.name, ref.protocol_version
        )

    def iter_dataset_refs(self, datasite_email: str) -> Iterator[DatasetRef]:
        root = self.config.public_datasets_root_for_datasite(datasite_email)
        if not root.exists():
            return
        for entry in sorted(p for p in root.iterdir() if p.is_dir()):
            # v<n>/ segments belong to a versioned codec, not the flat layout.
            if is_protocol_dir_name(entry.name):
                continue
            if (entry / self.metadata_marker).exists():
                yield DatasetRef(datasite_email, entry.name, "0")

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
