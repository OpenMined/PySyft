from pathlib import Path
from typing import Iterator

import yaml
from syft_migration import MigratableObject

from ..config import protocol_dir_name
from ..job_ref import JobRef
from .base import ProtocolCodec


class ProtocolCodecV1(ProtocolCodec):
    """Versioned layout (>= 0.1.39): jobs under <ds_email>/v<n>/, identity fields on disk."""

    version = "1"
    protocol_versions = ["1"]

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
            for protocol_version in self.protocol_versions:
                version_dir = ds_dir / protocol_dir_name(protocol_version)
                if not version_dir.is_dir():
                    continue
                for job_dir in sorted(p for p in version_dir.iterdir() if p.is_dir()):
                    if (job_dir / marker).exists():
                        yield JobRef(
                            datasite_email, ds_dir.name, job_dir.name, protocol_version
                        )

    def read(self, path: Path, canonical_name: str) -> dict:
        # Files already carry canonical_name/version on disk.
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def write(self, path: Path, obj: MigratableObject) -> None:
        data = obj.disk_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
