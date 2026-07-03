from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import yaml
from syft_migration import (
    BaseVersionsSchema,
    MigratableObject,
    MigrationRegistry,
    MigrationService,
    ProtocolSchema,
)

from .config import SyftJobConfig, is_protocol_dir_name
from .migrations.registry import JOB_PROTOCOL_VERSION, job_registry
from .models import JobState, JobSubmissionMetadata


@dataclass(frozen=True)
class JobRef:
    """One job on disk: who owns it, who submitted it, and its protocol layout."""

    datasite_email: str  # DO whose datasite holds the job
    ds_email: str  # submitter
    job_name: str
    protocol_version: str  # "0" (no path segment) or "1"+ (v<n> segment)


class JobManager:
    """All JobState/JobSubmissionMetadata filesystem IO and path resolution.

    Reads upgrade objects to the latest registered version in memory; writes
    downgrade them to what the reading peer understands. The protocol version
    of a discovered job comes from its on-disk layout; new submissions use the
    target peer's protocol version (current unless a peer schema says otherwise).
    """

    def __init__(
        self,
        config: SyftJobConfig,
        registry: MigrationRegistry = job_registry,
        peer_schemas: Optional[dict[str, ProtocolSchema]] = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.service = MigrationService(registry=registry)
        # peer email -> job ProtocolSchema; filled in by syft-client later.
        # Peers without an entry are assumed to run the current protocol.
        self.peer_schemas: dict[str, ProtocolSchema] = peer_schemas or {}

    # -- peers / protocol ----------------------------------------------------
    def protocol_version_for_peer(self, peer_email: str) -> str:
        schema = self.peer_schemas.get(peer_email)
        return schema.version if schema else JOB_PROTOCOL_VERSION

    def _write_schema(self, ref: JobRef, reader_email: str) -> BaseVersionsSchema:
        """The schema to downgrade to before writing for ``reader_email``.

        The job's own layout (``ref.protocol_version``) decides the path and
        format; a known peer schema only refines the object versions when it
        speaks the same protocol version.
        """
        schema = self.peer_schemas.get(reader_email)
        if schema is not None and schema.version == ref.protocol_version:
            return schema
        return self.registry.schema_for_protocol_version(ref.protocol_version)

    # -- path resolution -------------------------------------------------------
    def submission_dir(self, ref: JobRef) -> Path:
        return self.config.get_job_submission_dir(
            ref.datasite_email, ref.ds_email, ref.job_name, ref.protocol_version
        )

    def review_dir(self, ref: JobRef) -> Path:
        return self.config.get_review_job_dir(
            ref.datasite_email, ref.ds_email, ref.job_name, ref.protocol_version
        )

    def new_submission_ref(self, do_email: str, job_name: str) -> JobRef:
        """A ref for submitting a new job to ``do_email``."""
        return JobRef(
            datasite_email=do_email,
            ds_email=self.config.current_user_email,
            job_name=job_name,
            protocol_version=self.protocol_version_for_peer(do_email),
        )

    # -- naming ------------------------------------------------------------------
    @staticmethod
    def validate_job_name(job_name: str) -> None:
        if is_protocol_dir_name(job_name):
            raise ValueError(
                f"Job name {job_name!r} is reserved for protocol version directories"
            )

    # -- scanning (all protocol layouts) -------------------------------------------
    def iter_submission_refs(self, datasite_email: str) -> Iterator[JobRef]:
        yield from self._iter_refs(
            self.config.get_all_submissions_dir(datasite_email),
            datasite_email,
            marker="config.yaml",
        )

    def iter_review_refs(self, datasite_email: str) -> Iterator[JobRef]:
        yield from self._iter_refs(
            self.config.get_review_dir(datasite_email),
            datasite_email,
            marker="state.yaml",
        )

    def _iter_refs(
        self, root: Path, datasite_email: str, marker: str
    ) -> Iterator[JobRef]:
        if not root.exists():
            return
        for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for entry in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
                if is_protocol_dir_name(entry.name):
                    yield from self._refs_in_protocol_dir(
                        entry, datasite_email, ds_dir.name, marker
                    )
                elif (entry / marker).exists():
                    yield JobRef(datasite_email, ds_dir.name, entry.name, "0")

    def _refs_in_protocol_dir(
        self, protocol_dir: Path, datasite_email: str, ds_email: str, marker: str
    ) -> Iterator[JobRef]:
        protocol_version = protocol_dir.name.removeprefix("v")
        for job_dir in sorted(p for p in protocol_dir.iterdir() if p.is_dir()):
            if (job_dir / marker).exists():
                yield JobRef(datasite_email, ds_email, job_dir.name, protocol_version)

    def find_submission_ref(
        self, datasite_email: str, job_name: str, ds_email: Optional[str] = None
    ) -> JobRef:
        """The unique submission ref for ``job_name``; use ``ds_email`` to disambiguate."""
        matches = [
            ref
            for ref in self.iter_submission_refs(datasite_email)
            if ref.job_name == job_name
            and (ds_email is None or ref.ds_email == ds_email)
        ]
        if not matches:
            raise FileNotFoundError(f"Job '{job_name}' not found")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple jobs named '{job_name}' found; pass user to disambiguate"
            )
        return matches[0]

    # -- model IO ---------------------------------------------------------------
    def read_submission(self, ref: JobRef) -> JobSubmissionMetadata:
        """Load a submission's config.yaml, upgraded to the latest version."""
        extra = {"submitted_by": ref.ds_email, "datasite_email": ref.datasite_email}
        path = self.submission_dir(ref) / "config.yaml"
        return self._load_upgraded(path, "JobSubmissionMetadata", extra)

    def read_state(self, ref: JobRef) -> Optional[JobState]:
        """Load a job's state.yaml, upgraded to the latest version; None if absent."""
        path = self.review_dir(ref) / "state.yaml"
        if not path.exists():
            return None
        return self._load_upgraded(path, "JobState", {})

    def write_submission(self, ref: JobRef, metadata: JobSubmissionMetadata) -> Path:
        """Write config.yaml in the version/format the datasite owner can read."""
        path = self.submission_dir(ref) / "config.yaml"
        self._write_downgraded(path, metadata, ref, reader_email=ref.datasite_email)
        return path

    def write_state(self, ref: JobRef, state: JobState) -> Path:
        """Write state.yaml in the version/format the submitter can read."""
        path = self.review_dir(ref) / "state.yaml"
        self._write_downgraded(path, state, ref, reader_email=ref.ds_email)
        return path

    # -- internals -----------------------------------------------------------------
    def _load_upgraded(
        self, path: Path, canonical_name: str, extra: dict
    ) -> MigratableObject:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        data.update(extra)
        # Protocol-0 files predate the identity fields; they are all version 1.
        data.setdefault("canonical_name", canonical_name)
        data.setdefault("version", "1")
        obj = self.service.load(data)
        return self.service.migrate(obj, self.registry.latest_version(canonical_name))

    def _write_downgraded(
        self, path: Path, obj: MigratableObject, ref: JobRef, reader_email: str
    ) -> None:
        schema = self._write_schema(ref, reader_email=reader_email)
        downgraded = self.service.migrate_to_schema(obj, schema)
        data = downgraded.disk_dict()
        if ref.protocol_version == "0":
            # Byte-match the pre-versioning (<= 0.1.38) on-disk format.
            data.pop("canonical_name", None)
            data.pop("version", None)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
