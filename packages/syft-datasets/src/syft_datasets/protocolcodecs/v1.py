from pathlib import Path
from typing import Iterator

import yaml
from syft_migration import MigratableObject

from ..config import protocol_dir_name
from ..dataset_ref import DatasetRef
from .base import ProtocolCodec


class ProtocolCodecV1(ProtocolCodec):
    """Versioned layout (>= 0.1.21): datasets under syft_datasets/v<n>/, identity fields on disk."""

    version = "1"
    protocol_versions = ["1"]

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
        for protocol_version in self.protocol_versions:
            version_dir = root / protocol_dir_name(protocol_version)
            if not version_dir.is_dir():
                continue
            for entry in sorted(p for p in version_dir.iterdir() if p.is_dir()):
                if (entry / self.metadata_marker).exists():
                    yield DatasetRef(datasite_email, entry.name, protocol_version)

    def read(self, path: Path, canonical_name: str) -> dict:
        # Files already carry canonical_name/version on disk.
        return yaml.safe_load(path.read_text()) or {}

    def write(self, path: Path, obj: MigratableObject) -> None:
        data = obj.disk_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, indent=2, sort_keys=False))
