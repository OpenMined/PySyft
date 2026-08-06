from pathlib import Path
from typing import Iterator, Optional

from syft_migration import (
    MigratableObject,
    MigrationError,
    MigrationRegistry,
    MigrationService,
    ProtocolSchema,
)

from .config import SyftJobConfig, is_protocol_dir_name
from .job_ref import JobRef, JobStateNotFoundError
from .migrations.registry import JOB_PROTOCOL_VERSION, job_registry
from .models import JobState, JobSubmissionMetadata
from .protocolcodecs import CODECS, ProtocolCodec

__all__ = ["JobRef", "JobStateNotFoundError", "JobStorage"]


class JobStorage:
    """Job filesystem IO and path resolution, delegating disk layout to codecs.

    JobStorage owns migration: reads upgrade objects to the latest registered
    version in memory; writes downgrade them to what the reading peer understands.
    Each on-disk storage format lives behind a ProtocolCodec, selected by the
    protocol version (from a job's on-disk layout for reads; the negotiated
    version for new submissions). The last codec is the current protocol.
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
        # peer email -> job ProtocolSchema; syft-client passes PeerManager's
        # live map here (updated in place as peer version files load). Peers
        # without an entry are assumed to run the current protocol.
        # `is not None`, not `or`: syft-client passes a live (initially empty)
        # dict it mutates as peer version files load; `or {}` would drop the
        # shared reference and freeze negotiation at construction-time state.
        self.peer_schemas: dict[str, ProtocolSchema] = (
            peer_schemas if peer_schemas is not None else {}
        )
        self.codecs = [cls(config) for cls in CODECS]

    @property
    def _codec_by_protocol_version(self) -> dict[str, ProtocolCodec]:
        # protocol version -> codec; one codec may serve several versions.
        return {
            protocol_version: codec
            for codec in self.codecs
            for protocol_version in codec.protocol_versions
        }

    def _codec_for(self, protocol_version: str) -> ProtocolCodec:
        codec = self._codec_by_protocol_version.get(protocol_version)
        if codec is None:
            raise MigrationError(
                f"No codec for job protocol version {protocol_version!r}"
            )
        return codec

    # -- peers / protocol ----------------------------------------------------
    def negotiated_protocol_version_for_peer(
        self, peer_email: str, raise_on_unknown: bool = True
    ) -> str:
        """The job protocol version to speak with ``peer_email``.

        Negotiated as the minimum of our own protocol version and the peer's,
        so both sides use a version they can read. The result must also be at or
        above the floor of each side, or the negotiation raises. A peer without a
        known schema raises by default; with ``raise_on_unknown=False`` it is
        assumed to run the current protocol.
        """
        schema = self.peer_schemas.get(peer_email)
        if schema is not None:
            return self.registry.negotiate_protocol_version(
                peer_version=schema.version,
                peer_min=schema.min_supported_version,
            )
        if raise_on_unknown:
            raise MigrationError(
                f"No job protocol schema known for peer {peer_email!r}"
            )
        return JOB_PROTOCOL_VERSION

    def _get_write_target_schema(
        self, ref: JobRef, reader_email: str
    ) -> ProtocolSchema:
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
        return self._codec_for(ref.protocol_version).submission_dir(ref)

    def review_dir(self, ref: JobRef) -> Path:
        return self._codec_for(ref.protocol_version).review_dir(ref)

    def new_submission_ref(self, do_email: str, job_name: str) -> JobRef:
        """A ref for submitting a new job to ``do_email``."""
        return JobRef(
            datasite_email=do_email,
            ds_email=self.config.current_user_email,
            job_name=job_name,
            # Peers without a known schema are assumed to run the current
            # protocol.
            protocol_version=self.negotiated_protocol_version_for_peer(
                do_email, raise_on_unknown=False
            ),
        )

    # -- naming ------------------------------------------------------------------
    @staticmethod
    def validate_job_name(job_name: str) -> None:
        if is_protocol_dir_name(job_name):
            raise ValueError(
                f"Job name {job_name!r} is reserved for protocol version directories"
            )

    # -- scanning (union over all protocol layouts) -------------------------------
    def iter_submission_refs(self, datasite_email: str) -> Iterator[JobRef]:
        """Yield a ref per job under app_data/job/inbox/, across all codecs."""
        for codec in self.codecs:
            yield from codec.iter_submission_refs(datasite_email)

    def iter_review_refs(self, datasite_email: str) -> Iterator[JobRef]:
        """Yield a ref per job under app_data/job/review/, across all codecs."""
        for codec in self.codecs:
            yield from codec.iter_review_refs(datasite_email)

    def find_submission_ref(
        self, datasite_email: str, job_name: str, ds_email: Optional[str] = None
    ) -> JobRef:
        """The unique inbox/<ds_email>/[v<n>/]<job_name>/ ref; ``ds_email`` disambiguates."""
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
        codec = self._codec_for(ref.protocol_version)
        data = codec.read(codec.submission_metadata_path(ref), "JobSubmissionMetadata")
        # Identity comes from the path (JobRef), never the DS-writable file.
        data.update(
            {
                "submitted_by": ref.ds_email,
                "datasite_email": ref.datasite_email,
                "name": ref.job_name,
            }
        )
        return self._upgrade(data, "JobSubmissionMetadata")

    def read_state(self, ref: JobRef) -> JobState:
        """Load a job's state.yaml, upgraded to the latest version.

        Raises JobStateNotFoundError if the state.yaml does not exist.
        """
        codec = self._codec_for(ref.protocol_version)
        path = codec.state_path(ref)
        if not path.exists():
            raise JobStateNotFoundError(f"Job state not found for {ref.job_name}")
        return self._upgrade(codec.read(path, "JobState"), "JobState")

    def write_submission(self, ref: JobRef, metadata: JobSubmissionMetadata) -> Path:
        """Write config.yaml in the version/format the peer can read."""
        codec = self._codec_for(ref.protocol_version)
        return self._write(
            codec,
            codec.submission_metadata_path(ref),
            metadata,
            ref,
            reader_email=ref.datasite_email,
        )

    def write_state(self, ref: JobRef, state: JobState) -> Path:
        """Write state.yaml in the version/format the peer can read."""
        codec = self._codec_for(ref.protocol_version)
        return self._write(
            codec, codec.state_path(ref), state, ref, reader_email=ref.ds_email
        )

    # -- internals -----------------------------------------------------------------
    def _upgrade(self, data: dict, canonical_name: str) -> MigratableObject:
        obj = self.service.load(data)
        return self.service.migrate(obj, self.registry.latest_version(canonical_name))

    def _write(
        self,
        codec: ProtocolCodec,
        path: Path,
        obj: MigratableObject,
        ref: JobRef,
        reader_email: str,
    ) -> Path:
        """Downgrade ``obj`` to the reader's version, then let the codec persist it."""
        schema = self._get_write_target_schema(ref, reader_email=reader_email)
        downgraded = self.service.migrate_to_schema(obj, schema)
        codec.write(path, downgraded)
        return path
