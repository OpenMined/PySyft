from pathlib import Path
from typing import Iterator

import yaml
from syft_migration import MigratableObject

from ..config import is_protocol_dir_name
from ..job_ref import JobRef
from .base import ProtocolCodec


class ProtocolCodecV0(ProtocolCodec):
    """Pre-versioning layout (<= 0.1.38): jobs flat under <ds_email>/, no identity fields."""

    version = "0"
    protocol_versions = ["0"]

    def submission_dir(self, ref: JobRef) -> Path:
        return self.config.get_job_submission_dir(
            ref.datasite_email, ref.ds_email, ref.job_name, ref.protocol_version
        )

    def review_dir(self, ref: JobRef) -> Path:
        return self.config.get_review_job_dir(
            ref.datasite_email, ref.ds_email, ref.job_name, ref.protocol_version
        )

    def submission_metadata_path(self, ref: JobRef) -> Path:
        return self.submission_dir(ref) / self.submission_marker

    def state_path(self, ref: JobRef) -> Path:
        return self.review_dir(ref) / self.state_marker

    def iter_submission_refs(self, datasite_email: str) -> Iterator[JobRef]:
        root = self.config.get_all_submissions_dir(datasite_email)
        yield from self._scan(root, datasite_email, self.submission_marker)

    def iter_review_refs(self, datasite_email: str) -> Iterator[JobRef]:
        root = self.config.get_review_dir(datasite_email)
        yield from self._scan(root, datasite_email, self.state_marker)

    def _scan(self, root: Path, datasite_email: str, marker: str) -> Iterator[JobRef]:
        if not root.exists():
            return
        for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for entry in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
                # v<n>/ segments belong to a versioned codec, not the flat layout.
                if is_protocol_dir_name(entry.name):
                    continue
                if (entry / marker).exists():
                    yield JobRef(datasite_email, ds_dir.name, entry.name, "0")

    def read(self, path: Path, canonical_name: str) -> dict:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        # Protocol-0 files predate the identity fields; they are all version 1.
        data.setdefault("canonical_name", canonical_name)
        data.setdefault("version", "1")
        return data

    def write(self, path: Path, obj: MigratableObject) -> None:
        data = obj.disk_dict()
        # Byte-match the pre-versioning (<= 0.1.38) on-disk format.
        data.pop("canonical_name", None)
        data.pop("version", None)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
